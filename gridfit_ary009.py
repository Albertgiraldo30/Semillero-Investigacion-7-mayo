# -*- coding: utf-8 -*-
"""
Estimador de grilla ARY009 guiado por los SPOTS (no por 3 anclas).

Modelo rigido de 5 parametros:
    nodo(col, fila) = O + R(theta) . ( T_COLS[col]*W , (fila-1.5)*rs )
  O=(x0,y0) centro del par A1/A2, W = ancho total A->E, rs = paso de fila,
  theta = inclinacion. La topologia (5 columnas con fracciones fijas, 24 filas)
  nunca cambia: la grilla solo puede trasladarse, escalar en bloque y rotar,
  nunca deformarse para perseguir spots sueltos.

Etapas:
  A) theta  : angulo que maximiza la picudez de la proyeccion de centroides.
  B) rs, y0 : filtro peine de 24 dientes sobre la proyeccion Y (variante por
              autocorrelacion disponible para comparar).
  C) W, x0  : filtro peine de 5 dientes con las fracciones T_COLS.
  D) refino : pattern-search conjunto sobre los 5 parametros, coste robusto
              SPOT->NODO con umbral FIJO en pixeles (si tau escalara con rs,
              encoger la grilla abarataria el coste = colapso).
"""
import numpy as np

from mapa_array import T_COLS  # geometria del kit: fuente unica

# (si esta linea falla, mapa_array.py no esta en el path: este modulo
#  esta pensado para vivir junto al resto del proyecto ARY009)
TC = np.array([T_COLS[c] for c in "ABCDE"], dtype=np.float64)
FILAS = np.arange(1, 25, dtype=np.float64)
FREF_TOP, FREF_BOT = 1.5, 23.5


def _pesos(spots):
    a = np.array([s[2] for s in spots], dtype=np.float64)
    w = np.sqrt(np.clip(a, 1.0, None))
    return w / w.max()


def _perfil(vals, w, lo, hi, sigma):
    n = int(np.ceil(hi - lo)) + 1
    if n < 4:
        return np.zeros(4), lo
    idx = np.arange(n, dtype=np.float64) + lo
    d = (vals[:, None] - idx[None, :]) / sigma
    p = (w[:, None] * np.exp(-0.5 * d * d)).sum(axis=0)
    return p, lo


def _picudez(p):
    s = p.sum()
    if s <= 0:
        return 0.0
    q = p / s
    return float((q * q).sum() * len(q))


def _rot(xy, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([xy[:, 0] * c + xy[:, 1] * s,
                     -xy[:, 0] * s + xy[:, 1] * c], axis=1)


def estimar_angulo(xy, w, rs_ap, grados=15.0, paso=0.25):
    mejor, best = 0.0, -1.0
    sig = max(0.8, rs_ap * 0.14)
    cen = xy.mean(axis=0)
    p = xy - cen
    for g in np.arange(-grados, grados + 1e-9, paso):
        th = np.radians(g)
        r = _rot(p, th)
        py, _ = _perfil(r[:, 1], w, r[:, 1].min() - 2, r[:, 1].max() + 2, sig)
        px, _ = _perfil(r[:, 0], w, r[:, 0].min() - 2, r[:, 0].max() + 2, sig)
        sc = _picudez(py) + 0.5 * _picudez(px)
        if sc > best:
            best, mejor = sc, th
    return mejor, cen


def rs_por_autocorrelacion(yv, w, lo, hi, sigma):
    p, off = _perfil(yv, w, yv.min() - 2, yv.max() + 2, sigma)
    p = p - p.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    l0, l1 = int(max(2, lo)), int(min(len(ac) - 1, hi))
    if l1 <= l0:
        return None
    k = int(np.argmax(ac[l0:l1 + 1]))
    i = float(l0 + k)
    j = l0 + k
    if 0 < j < len(ac) - 1:
        a, b, c = ac[j - 1], ac[j], ac[j + 1]
        den = a - 2 * b + c
        if den != 0:
            i = j - 0.5 * (a - c) / den
    return i


def peine_filas(yv, w, rs_lo, rs_hi, n_rs=160):
    sig_f = 0.18
    mejor = (None, None, -1.0)
    for rs in np.linspace(rs_lo, rs_hi, n_rs):
        sig = rs * sig_f
        for ph in np.linspace(0.0, rs, max(8, int(rs * 2)), endpoint=False):
            y0 = yv.min() + ph
            t = (yv - y0) / rs
            r = np.round(t)
            d = np.abs(t - r) * rs
            ok = (r >= -1) & (r <= 24)
            sc = float((w * np.exp(-0.5 * (d / sig) ** 2) * ok).sum())
            if sc > mejor[2]:
                mejor = (rs, y0, sc)
    return mejor[0], mejor[1]


def peine_columnas(xv, w, W_lo, W_hi, n_W=140, signos=(-1.0, 1.0)):
    """Peine de 5 dientes con las fracciones T_COLS. Prueba las DOS
    orientaciones: en el ARY009 la columna A queda a la derecha (W<0), pero
    una membrana escaneada al reves daria W>0, y el coste lo decide."""
    mejor = (None, None, -1.0)
    xlo, xhi = xv.min(), xv.max()
    for signo in signos:
        for Wm in np.linspace(W_lo, W_hi, n_W):
            W = signo * Wm
            sig = max(0.7, Wm * 0.025)
            for x0 in np.linspace(xlo - 0.30 * Wm, xhi + 0.30 * Wm, 70):
                lines = x0 + TC * W
                d = np.abs(xv[:, None] - lines[None, :]).min(axis=1)
                sc = float((w * np.exp(-0.5 * (d / sig) ** 2)).sum())
                if sc > mejor[2]:
                    mejor = (W, x0, sc)
    return mejor[0], mejor[1]


def nodos(par):
    x0, y0, W, rs, th = par
    lx = np.repeat((TC * W)[:, None], 24, axis=1).ravel()
    ly = np.repeat(((FILAS - FREF_TOP) * rs)[None, :], 5, axis=0).ravel()
    c, s = np.cos(th), np.sin(th)
    return np.stack([x0 + lx * c - ly * s, y0 + lx * s + ly * c], axis=1)


def coste(par, sxy, w, tau):
    n = nodos(par)
    d2 = ((sxy[:, None, :] - n[None, :, :]) ** 2).sum(-1).min(axis=1)
    return float((w * np.minimum(d2, tau * tau)).sum())


def _proyectar(par, ratio_lo, ratio_hi, signo_fijo):
    """Mantiene el ajuste dentro de la fisica del kit ARY009.

    Sin esto el optimizador degenera: encogiendo W mete las 5 columnas mas
    juntas, cualquier spot cae cerca de ALGUN nodo y el coste baja sin que la
    grilla este mejor puesta. La relacion col_spacing/row_spacing del kit es
    ~1.35 en todas las membranas bien ancladas, asi que se acota.

    `signo_fijo` (-1 = A a la derecha, +1 = A a la izquierda, 0 = libre) fija
    la orientacion SIN mover la grilla: el espejo NO es una simetria de este
    modelo (T_COLS 0/.21/.43/.64/1 es asimetrico), asi que reflejar un ajuste
    ya hecho lo estropea. Por eso la orientacion se fija ANTES de optimizar y
    cada orientacion se ajusta por separado.
    """
    x0, y0, W, rs, th = par
    rs = max(1.0, rs)
    cs = abs(W) / 4.0
    cs = min(max(cs, ratio_lo * rs), ratio_hi * rs)
    s = (signo_fijo if signo_fijo else (1.0 if W >= 0 else -1.0))
    W = s * cs * 4.0
    return [x0, y0, W, rs, th]


def refinar(par, sxy, w, tau, pasos, ratio_lo=0.0, ratio_hi=9.9, signo_fijo=0,
            libres=(0, 1, 2, 3, 4)):
    acotar = (ratio_lo > 0.0 or ratio_hi < 9.9 or signo_fijo)
    par = _proyectar(par, ratio_lo, ratio_hi, signo_fijo) if acotar else list(par)
    c = coste(par, sxy, w, tau)
    pasos = list(pasos)
    for _ in range(120):
        mejoro = False
        for i in libres:
            for sg in (1, -1):
                cand = list(par)
                cand[i] += sg * pasos[i]
                if i == 3 and acotar:          # rs cambia -> reescalar W
                    cand[2] = cand[2] * (cand[3] / par[3]) if par[3] else cand[2]
                if acotar:
                    cand = _proyectar(cand, ratio_lo, ratio_hi, signo_fijo)
                cc = coste(cand, sxy, w, tau)
                if cc < c - 1e-9:
                    par, c, mejoro = cand, cc, True
                    break
        if not mejoro:
            pasos = [p * 0.5 for p in pasos]
            if max(pasos) < 0.02:
                break
    return par, c


def ajustar(spots, shape, rs_prior=None, metodo_rs="peine", angulo=True,
            refino=True, ratio_lo=0.0, ratio_hi=9.9, signo=False):
    h, wid = shape[:2]
    if len(spots) < 8:
        return None, {"motivo": "pocos spots"}
    sxy = np.array([[s[0], s[1]] for s in spots], dtype=np.float64)
    w = _pesos(spots)
    rs_ap = rs_prior if rs_prior else h / 24.0

    if angulo:
        th, cen = estimar_angulo(sxy, w, rs_ap)
    else:
        th, cen = 0.0, sxy.mean(axis=0)
    r = _rot(sxy - cen, th)

    rs_lo, rs_hi = h / 34.0, h / 17.0
    if metodo_rs == "autocorr":
        sig = max(0.8, rs_ap * 0.14)
        rs = rs_por_autocorrelacion(r[:, 1], w, rs_lo, rs_hi, sig)
        if rs is None:
            return None, {"motivo": "autocorr sin pico"}
        _, y0r = peine_filas(r[:, 1], w, rs, rs, n_rs=1)
    else:
        rs, y0r = peine_filas(r[:, 1], w, rs_lo, rs_hi)
    if rs is None:
        return None, {"motivo": "sin paso de fila"}

    W_lo = 4 * rs * (ratio_lo if ratio_lo > 0 else 0.90)
    W_hi = 4 * rs * (ratio_hi if ratio_hi < 9.9 else 2.20)
    sg = (-1.0,) if signo == -1 else ((1.0,) if signo == 1 else (-1.0, 1.0))
    W, x0r = peine_columnas(r[:, 0], w, W_lo, W_hi, signos=sg)

    tau = 0.45 * rs
    c_, s_ = np.cos(th), np.sin(th)
    mejor = None
    for k in range(-3, 4):
        yk = y0r + k * rs
        px = cen[0] + x0r * c_ - yk * s_
        py = cen[1] + x0r * s_ + yk * c_
        par = [px, py, W, rs, th]
        cc = coste(par, sxy, w, tau)
        if mejor is None or cc < mejor[1]:
            mejor = (par, cc)
    par, c = mejor
    if refino:
        par, c = refinar(par, sxy, w, tau,
                         [rs * 0.35, rs * 0.35, rs * 0.30, rs * 0.06,
                          np.radians(1.2)],
                         ratio_lo=ratio_lo, ratio_hi=ratio_hi,
                         signo_fijo=signo)
    return par, {"theta_deg": float(np.degrees(par[4])), "rs": par[3],
                 "cs": par[2] / 4.0, "coste": c}


def par_a_refs(par):
    x0, y0, W, rs, th = par
    c, s = np.cos(th), np.sin(th)

    def P(tc, f):
        lx, ly = tc * W, (f - FREF_TOP) * rs
        return (x0 + lx * c - ly * s, y0 + lx * s + ly * c)

    return P(0.0, FREF_TOP), P(1.0, FREF_TOP), P(0.0, FREF_BOT)


# ---------------------------------------------------------------------------
# Desambiguacion de orientacion.
#
# El ajuste geometrico no distingue una grilla de su espejo ni de su volteo
# vertical: las 5 columnas y las 24 filas encajan igual de bien. Pero el kit
# ARY009 tiene Reference Spots en SOLO 3 de las 4 esquinas (A1/A2 arriba-dcha,
# A23/A24 abajo-dcha, E1/E2 arriba-izq) y un PBS vacio en D23/D24. Esa firma
# asimetrica fija la orientacion de forma unica: es para lo que esta puesta.
# ---------------------------------------------------------------------------
REF_NODOS = [('A', 1), ('A', 2), ('A', 23), ('A', 24), ('E', 1), ('E', 2)]
VACIO_NODOS = [('E', 23), ('E', 24), ('D', 23), ('D', 24)]


def _nodo(par, col, fila):
    x0, y0, W, rs, th = par
    c, s = np.cos(th), np.sin(th)
    lx, ly = T_COLS[col] * W, (fila - FREF_TOP) * rs
    return x0 + lx * c - ly * s, y0 + lx * s + ly * c


def _muestra(inv, x, y, r):
    h, w = inv.shape
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(inv[y0:y1, x0:x1].mean())


def variantes_orientacion(par):
    """Las 4 poses indistinguibles geometricamente: espejo X x volteo Y."""
    x0, y0, W, rs, th = par
    c, s = np.cos(th), np.sin(th)
    out = []
    for esp in (False, True):
        for vol in (False, True):
            X, Y, WW = x0, y0, W
            if esp:                                    # A pasa al otro extremo
                X, Y = X + WW * c, Y + WW * s
                WW = -WW
            if vol:                                    # fila 1 pasa abajo
                # el origen se muda al nodo de la fila 23.5 y las filas se
                # recorren al reves (rs negativo)
                dy = (FREF_BOT - FREF_TOP) * rs
                X, Y = X - dy * s, Y + dy * c
                out.append(([X, Y, WW, -rs, th], (esp, vol)))
                continue
            out.append(([X, Y, WW, rs, th], (esp, vol)))
    return out


def dentro(par, shape, frac=0.80):
    """Fraccion de nodos dentro del strip; una pose volteada que se sale de la
    membrana puntuaria 0 en todo y ganaria por defecto si no se descarta."""
    h, w = shape[:2]
    n = nodos(par)
    ok = ((n[:, 0] >= 0) & (n[:, 0] < w) & (n[:, 1] >= 0) & (n[:, 1] < h))
    return float(ok.mean()) >= frac


def puntaje_referencias(par, inv, r=None):
    """Firma de las 3 esquinas: refs brillantes, esquina vacia y PBS oscuros."""
    rs = abs(par[3])
    if r is None:
        r = max(2, int(rs * 0.35))
    ref = [_muestra(inv, *_nodo(par, c, f), r) for c, f in REF_NODOS]
    vac = [_muestra(inv, *_nodo(par, c, f), r) for c, f in VACIO_NODOS]
    fondo = float(np.median(inv))
    return float(np.mean(ref) - np.mean(vac)), float(np.mean(ref) - fondo)


def orientar(par, strip_gris):
    """Devuelve (par_orientado, info) eligiendo la pose con mejor firma."""
    maxv = np.iinfo(strip_gris.dtype).max if strip_gris.dtype.kind == 'u' else 255
    inv = (maxv - strip_gris) if strip_gris.mean() > maxv * 0.35 else strip_gris
    inv = inv.astype(np.float32)
    mejor, puntos = None, []
    for cand, etiqueta in variantes_orientacion(par):
        if not dentro(cand, strip_gris.shape):
            puntos.append((float('-inf'), etiqueta))
            continue
        sc, _ = puntaje_referencias(cand, inv)
        puntos.append((sc, etiqueta))
        if mejor is None or sc > mejor[0]:
            mejor = (sc, cand, etiqueta)
    if mejor is None:
        return par, {"score": float('nan'), "pose": None, "todos": puntos}
    return mejor[1], {"score": mejor[0], "pose": mejor[2], "todos": puntos}


# ---------------------------------------------------------------------------
# API definitiva: ajuste + orientacion.
#
# El espejo NO es simetria del modelo, asi que no se puede ajustar primero y
# reflejar despues. Se ajusta una grilla POR orientacion de columna (A a la
# derecha y A a la izquierda), se les anade el volteo de filas (ese SI es
# simetria, las 24 filas son equiespaciadas) y de las 4 poses ya optimizadas
# gana la que encaja con la firma de 3 esquinas del kit.
# ---------------------------------------------------------------------------
def ajustar_ary009(spots, strip_gris, rs_prior=None,
                   ratio_lo=1.25, ratio_hi=1.45, tol_coste=1.20):
    if len(spots) < 8:
        return None, {"motivo": "pocos spots (<8)"}
    sxy = np.array([[s[0], s[1]] for s in spots], dtype=np.float64)
    w = _pesos(spots)

    cands = []
    for sg in (-1, 1):
        par, info = ajustar(spots, strip_gris.shape, rs_prior=rs_prior,
                            ratio_lo=ratio_lo, ratio_hi=ratio_hi, signo=sg)
        if par is None:
            continue
        tau = 0.45 * par[3]
        cands.append((par, coste(par, sxy, w, tau), info))
    if not cands:
        return None, {"motivo": "sin ajuste"}

    c_min = min(c for _, c, _ in cands)
    maxv = np.iinfo(strip_gris.dtype).max if strip_gris.dtype.kind == "u" else 255
    inv = (maxv - strip_gris) if strip_gris.mean() > maxv * 0.35 else strip_gris
    inv = inv.astype(np.float32)

    mejor, todos = None, []
    for par, c, _ in cands:
        # solo el volteo de filas es simetria legitima del ajuste
        for vol in (False, True):
            cand = list(par)
            if vol:
                rs, th = par[3], par[4]
                dy = (FREF_BOT - FREF_TOP) * rs
                cand = [par[0] - dy * np.sin(th), par[1] + dy * np.cos(th),
                        par[2], -rs, th]
            if not dentro(cand, strip_gris.shape):
                continue
            sc, _ = puntaje_referencias(cand, inv)
            # una pose con geometria claramente peor no entra aunque su firma
            # salga alta: primero encajar, luego decidir el norte
            penal = c > c_min * tol_coste
            todos.append((float(sc), par[2] < 0, vol, bool(penal), float(c)))
            if penal:
                continue
            if mejor is None or sc > mejor[0]:
                mejor = (sc, cand)
    if mejor is None:
        par = min(cands, key=lambda t: t[1])[0]
        return par, {"pose": "fallback", "todos": todos}
    par = mejor[1]
    return par, {"score": mejor[0], "A_derecha": par[2] < 0,
                 "fila1_arriba": par[3] > 0, "rs": abs(par[3]),
                 "cs": abs(par[2]) / 4.0, "theta_deg": float(np.degrees(par[4])),
                 "todos": todos}


# ---------------------------------------------------------------------------
# GANADOR. Reparto de trabajo entre las dos fuentes de informacion:
#
#   - las 3 anclas de Reference Spots dicen la ORIENTACION (que lado es la
#     columna A, que punta es la fila 1). Eso es una etiqueta, y las anclas
#     aciertan en 22/23 tirillas.
#   - los spots dicen la GEOMETRIA (paso de fila, ancho, desplazamiento,
#     inclinacion). Eso es lo que las anclas hacen mal: toda la grilla cuelga
#     de 3 puntos y un error de 2 px en uno de ellos desplaza las 78
#     posiciones.
#
# Medido: el ajuste guiado por spots SOLO (sin anclas) gana en geometria pero
# se equivoca de espejo en 5/24 tirillas, y un espejo intercambia proteinas.
# Tomando la orientacion de las anclas se conservan las dos cosas.
# ---------------------------------------------------------------------------
def ajustar_con_ancla(spots, strip_gris, refs, rs_prior=None,
                      ratio_lo=1.25, ratio_hi=1.45):
    """refs = (tl, tr, bl) de encontrar_referencias(), o None."""
    if refs is None:
        return ajustar_ary009(spots, strip_gris, rs_prior=rs_prior,
                              ratio_lo=ratio_lo, ratio_hi=ratio_hi)
    tl, tr, bl = refs
    if rs_prior is None:
        rs_prior = max(2.0, abs(bl[1] - tl[1]) / 22.0)
    signo = -1 if (tr[0] - tl[0]) < 0 else 1
    par, info = ajustar(spots, strip_gris.shape, rs_prior=rs_prior,
                        ratio_lo=ratio_lo, ratio_hi=ratio_hi, signo=signo)
    if par is None:
        return None, {"motivo": info.get("motivo", "sin ajuste")}

    # volteo de filas: el que deja las 3 anclas mas cerca de sus nodos
    anc = np.array([tl, tr, bl], dtype=np.float64)
    mejor = None
    for vol in (False, True):
        cand = list(par)
        if vol:
            dy = (FREF_BOT - FREF_TOP) * par[3]
            cand = [par[0] - dy * np.sin(par[4]), par[1] + dy * np.cos(par[4]),
                    par[2], -par[3], par[4]]
        p = np.array([_nodo(cand, 'A', FREF_TOP), _nodo(cand, 'E', FREF_TOP),
                      _nodo(cand, 'A', FREF_BOT)], dtype=np.float64)
        d = float(np.sqrt(((p - anc) ** 2).sum(axis=1)).sum())
        if mejor is None or d < mejor[0]:
            mejor = (d, cand, vol)
    par = mejor[1]
    return par, {"A_derecha": par[2] < 0, "fila1_arriba": par[3] > 0,
                 "rs": abs(par[3]), "cs": abs(par[2]) / 4.0,
                 "theta_deg": float(np.degrees(par[4])),
                 "dist_anclas": mejor[0], "volteo": mejor[2]}
