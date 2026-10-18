# -*- coding: utf-8 -*-
"""
regresion.py - METRICA OFICIAL de deteccion de tirillas del analizador ARY009.

Corre el motor sobre los 6 .scn de fotos/ (24 tirillas en total) y cuenta
cuantas entregan intensidades. Guarda imagenes debug en tests/debug/ para que
se pueda verificar a ojo que la grilla cae sobre los spots.

REGLA DEL PROYECTO: una tirilla vacia es mejor que una grilla equivocada.
Por eso el script no premia solo el conteo: tambien reporta la relacion de
celda (col_spacing/row_spacing, ~1.3 en membranas reales) de cada tirilla
aceptada, que es el validador que delata una grilla mal puesta.

USO
    python tests/regresion.py                  # corre y compara vs linea base
    python tests/regresion.py --json out.json  # guarda resultado crudo
    python tests/regresion.py --quiet          # silencia el log del motor

SALIDA
    codigo 0 si no hay REGRESION (ninguna tirilla de la linea base se perdio)
    codigo 1 si alguna tirilla que antes funcionaba dejo de funcionar
"""
import os
import sys
import json
import argparse
import logging
import traceback

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# LINEA BASE medida el 2026-10-17 (16/24 tirillas con datos).
# NO se toca para "pasar" el test: es el contrato contra el que se compara.
# Las dos membranas marcadas PROTEGIDA van 4/4 y no pueden bajar nunca.
# ---------------------------------------------------------------------------
LINEA_BASE = {
    "C.mama -Curcuma":            2,
    "Linea SW480 colon-mango":    4,   # PROTEGIDA
    "Lineas de cancer de mama":   1,
    "SW480-Aspirina":             3,
    "Sw480- agraz":               2,
    "Sw620 colon-mango":          4,   # PROTEGIDA
}
PROTEGIDAS = {"Linea SW480 colon-mango", "Sw620 colon-mango"}
N_STRIPS = 4
TOTAL_TIRILLAS = len(LINEA_BASE) * N_STRIPS

# Rango de relacion de celda que se considera sano. Fuera de aqui la tirilla
# se reporta como sospechosa aunque haya entregado numeros.
RATIO_SANO = (1.15, 1.55)

# Los 6 Reference Spots, en orden fijo. Viven en 3 esquinas (A1/A2 sup-der,
# A23/A24 inf-der, E1/E2 sup-izq): esa asimetria es la que fija la orientacion.
COORDS_REF_ORD = (("A", 1), ("A", 2), ("A", 23), ("A", 24), ("E", 1), ("E", 2))
# Los 3 pares de duplicados de Reference. Cada par es la MISMA proteina en
# dos spots verticales, asi que sus dos lecturas deben parecerse.
PARES_REF = ((("A", 1), ("A", 2)), (("A", 23), ("A", 24)), (("E", 1), ("E", 2)))

# Piso de la firma de grilla. Medido sobre las 24 tirillas: las bien puestas
# dan equilibrio por par 0.72-0.99 y 0 nodos colapsados; las rotas, ~0.25.
BAL_PAR_PISO = 0.45

DIR_FOTOS = os.path.join(RAIZ, "fotos")
DIR_DEBUG = os.path.join(RAIZ, "tests", "debug")


def _ratio_celda(refs):
    """(col_spacing, row_spacing, ratio) de unas anclas (tl, tr, bl)."""
    if refs is None:
        return None, None, None
    tl, tr, bl = refs
    cs = abs(tr[0] - tl[0]) / 4.0
    rs = abs(bl[1] - tl[1]) / 22.0
    return cs, rs, (cs / rs if rs > 0 else 0.0)


def _muestra(inv, x, y, r):
    h, w = inv.shape
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(inv[y0:y1, x0:x1].mean())


def _firma_grilla(sv):
    """Dos chivatos que la relacion de celda NO puede ver.

    El ratio solo mide la FORMA de la celda: una grilla con forma perfecta
    puede estar corrida media fila y el ratio sale sano igual. Estos dos
    indicadores atrapan ese caso, y son los que dejaron pasar a Sw620 M2
    (ratio 1.378, "sana", con el Reference A23 desplomado y el PBS 6x
    contaminado):

      bal_par      = el PEOR equilibrio dentro de un par de duplicados de
                     Reference (A1/A2, A23/A24, E1/E2): min/max de los dos.
                     Los duplicados son la MISMA proteina, asi que con la
                     grilla bien puesta leen parecido (0.72-0.99 medido). Si
                     el ancla aterriza en UNO de los dos spots en vez de en el
                     centro del par, su companero se desploma y esto cae a
                     ~0.25.
                     Se mide por PAR y no min/max entre los 6 a proposito: hay
                     membranas con la columna A entera tenue (C.mama M2) y
                     tirillas donde la caja de muestreo de E1/E2 se sale por
                     el borde del recorte (Linea SW480 M1). En las dos el
                     min/max global se desploma sin que la grilla este mal; el
                     equilibrio por par las distingue.
      colapsados   = pares de filas de la misma columna a menos de 1.5 px. Una
                     grilla rigida bien ajustada da 0. Dos filas en el mismo
                     pixel significan que un duplicado se mide dos veces y la
                     otra fila se pierde.
      pbs          = senal del control negativo D23/D24, que debe leer vacio.
                     Se resta como fondo a las 35 proteinas, asi que un PBS
                     sobre senal real envenena la tirilla completa.
    """
    pos = sv.get("posiciones_cache")
    strip = sv.get("strip_pos")
    if not pos or strip is None or strip.size == 0:
        return None
    inv = 255.0 - strip.astype(np.float32)
    r = sv.get("r_cache") or 4
    bal, peor_par = 1.0, None
    for a, b in PARES_REF:
        if a not in pos or b not in pos:
            continue
        va, vb = _muestra(inv, *pos[a], r), _muestra(inv, *pos[b], r)
        if max(va, vb) <= 0:
            continue
        q = min(va, vb) / max(va, vb)
        if q < bal:
            bal, peor_par = q, f"{a[0]}{a[1]}/{b[0]}{b[1]}"
    pbs = [_muestra(inv, *pos[c], r)
           for c in (("D", 23), ("D", 24)) if c in pos]
    col_ys = {}
    for (col, fila), (px, py) in pos.items():
        col_ys.setdefault(col, []).append((fila, py))
    colapsados = 0
    for lst in col_ys.values():
        lst.sort()
        colapsados += sum(1 for a, b in zip(lst, lst[1:])
                          if abs(b[1] - a[1]) < 1.5)
    return {
        "bal_par": round(bal, 3),
        "peor_par": peor_par,
        "colapsados": colapsados,
        "pbs": round(max(pbs), 1) if pbs else None,
        "fondo": round(float(np.median(inv)), 1),
    }


def _guardar_recortes(motor, nombre_base):
    """Un PNG por tirilla con su grilla dibujada, ampliado 3x.

    El debug de imagen completa sirve para ver el reparto de membranas, pero
    para juzgar si la grilla cae EN los spots hace falta zoom por tirilla.
    """
    from motor_apoptosis import MAPA_PROTEINAS, PRO_APOPTOTICAS, ANTI_APOPTOTICAS

    rutas = []
    for i, sv in enumerate(getattr(motor, "_strips_visual", []) or []):
        strip = sv.get("strip_img")
        if strip is None or strip.size == 0:
            continue
        vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        refs = sv.get("refs")
        pos = sv.get("posiciones_cache")
        r = sv.get("r_cache") or 4
        if refs is not None:
            for (px, py), lbl in zip(refs, ("TL", "TR", "BL")):
                cv2.circle(vis, (int(px), int(py)), r + 4, (0, 140, 255), 1)
                cv2.putText(vis, lbl, (int(px) + r + 2, int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)
        if pos:
            for (col, fila), (px, py) in pos.items():
                prot = MAPA_PROTEINAS.get((col, fila), "?")
                if prot == "Reference":
                    color = (0, 215, 255)
                elif prot == "PBS":
                    color = (200, 180, 0)
                elif prot in PRO_APOPTOTICAS:
                    color = (0, 0, 200)
                elif prot in ANTI_APOPTOTICAS:
                    color = (200, 0, 0)
                else:
                    color = (0, 180, 0)
                cv2.rectangle(vis, (px - r, py - r), (px + r, py + r), color, 1)
        vis = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        etiqueta = "CON-DATOS" if refs is not None else "VACIA"
        cv2.putText(vis, f"M{i+1} {etiqueta}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ruta = os.path.join(DIR_DEBUG, f"{nombre_base}__M{i+1}.png")
        cv2.imwrite(ruta, vis)
        rutas.append(ruta)
    return rutas


def analizar_archivo(ruta_scn, quiet=True):
    """Devuelve el dict de resultado de una membrana (4 tirillas)."""
    from motor_apoptosis import AnalizadorGrillaARY009

    nombre = os.path.splitext(os.path.basename(ruta_scn))[0]
    res = {
        "archivo": nombre,
        "con_datos": 0,
        "esperado": LINEA_BASE.get(nombre),
        "protegida": nombre in PROTEGIDAS,
        "tirillas": [],
        "error": None,
    }
    motor = AnalizadorGrillaARY009()
    ruta_debug = os.path.join(DIR_DEBUG, f"{nombre}__full.png")
    try:
        intensidades = motor.analizar_imagen(ruta_scn, n_strips=N_STRIPS,
                                             ruta_debug=ruta_debug)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        if not quiet:
            traceback.print_exc()
        return res

    visual = getattr(motor, "_strips_visual", []) or []

    # Cuantas tirillas logro SEPARAR la imagen. Es un indicador distinto del
    # conteo de tirillas con datos y hay que vigilarlo aparte: las 6 membranas
    # tienen 4 tirillas fisicas, asi que separar menos de 4 es un techo
    # estructural — esa tirilla ya no se intenta analizar, no es que se haya
    # rechazado por mala.
    res["separadas"] = len(visual)
    if len(visual) < N_STRIPS:
        res["techo"] = f"solo {len(visual)}/{N_STRIPS} tirillas separadas"

    for i, inten in enumerate(intensidades):
        sv = visual[i] if i < len(visual) else {}
        refs = sv.get("refs")
        cs, rs, ratio = _ratio_celda(refs)
        n_prot = sum(1 for v in (inten or {}).values()
                     if isinstance(v, (int, float)))
        res["tirillas"].append({
            "strip": i + 1,
            "con_datos": bool(inten),
            "n_proteinas": n_prot,
            "ratio_celda": round(ratio, 3) if ratio else None,
            "col_spacing": round(cs, 1) if cs else None,
            "row_spacing": round(rs, 1) if rs else None,
            "alertas": sorted((sv.get("alertas") or {}).keys()),
            "firma": _firma_grilla(sv) if inten else None,
        })
        if inten:
            res["con_datos"] += 1

    res["recortes"] = _guardar_recortes(motor, nombre)
    res["debug_full"] = ruta_debug
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="ruta donde volcar el resultado crudo")
    ap.add_argument("--quiet", action="store_true", help="silencia el log del motor")
    ap.add_argument("--solo", help="analiza un solo archivo (nombre sin .scn)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="    %(levelname)s %(message)s",
    )
    if args.quiet:
        logging.getLogger("motor_apoptosis").setLevel(logging.ERROR)

    os.makedirs(DIR_DEBUG, exist_ok=True)

    scns = sorted(f for f in os.listdir(DIR_FOTOS) if f.lower().endswith(".scn"))
    if args.solo:
        scns = [f for f in scns if os.path.splitext(f)[0] == args.solo]
    if not scns:
        print("No hay .scn en fotos/")
        return 2

    resultados = []
    for f in scns:
        nombre = os.path.splitext(f)[0]
        print(f"\n=== {nombre} ===")
        r = analizar_archivo(os.path.join(DIR_FOTOS, f), quiet=args.quiet)
        resultados.append(r)
        if r["error"]:
            print(f"  ERROR: {r['error']}")
            continue
        for t in r["tirillas"]:
            if t["con_datos"]:
                fi = t.get("firma") or {}
                mal = (fi.get("bal_par") is not None
                       and fi["bal_par"] < BAL_PAR_PISO) \
                    or fi.get("colapsados", 0) > 0
                marca = "  <-- FIRMA ROTA" if mal else ""
                print(f"  M{t['strip']}  OK    {t['n_proteinas']:>2} proteinas  "
                      f"celda {t['col_spacing']}x{t['row_spacing']}px "
                      f"ratio={t['ratio_celda']}  "
                      f"bal={fi.get('bal_par')}({fi.get('peor_par')}) colaps={fi.get('colapsados')} "
                      f"pbs={fi.get('pbs')}/fondo={fi.get('fondo')}{marca}")
            else:
                print(f"  M{t['strip']}  vacia")
        aviso = f"   <-- {r['techo']}" if r.get("techo") else ""
        print(f"  -> {r['con_datos']}/{N_STRIPS} con datos, "
              f"{r.get('separadas', 0)}/{N_STRIPS} separadas{aviso}")

    # ---------------- tabla comparativa ----------------
    print("\n" + "=" * 78)
    print(f"{'MEMBRANA':<30}{'BASE':>6}{'AHORA':>7}{'DELTA':>7}{'SEP':>5}   ESTADO")
    print("-" * 78)
    total_base = total_ahora = total_sep = 0
    regresiones, mejoras = [], []
    for r in resultados:
        base, ahora = r["esperado"], r["con_datos"]
        sep = r.get("separadas", 0)
        total_sep += sep
        if base is None:
            print(f"{r['archivo']:<30}{'-':>6}{ahora:>7}{'-':>7}{sep:>5}   (fuera de linea base)")
            continue
        total_base += base
        total_ahora += ahora
        d = ahora - base
        if d < 0:
            estado = "REGRESION" + ("  !! PROTEGIDA !!" if r["protegida"] else "")
            regresiones.append((r["archivo"], base, ahora, r["protegida"]))
        elif d > 0:
            estado = "mejora"
            mejoras.append((r["archivo"], base, ahora))
        else:
            estado = "igual"
        print(f"{r['archivo']:<30}{base:>6}{ahora:>7}{d:>+7}{sep:>5}   {estado}")
    print("-" * 78)
    print(f"{'TOTAL':<30}{total_base:>6}{total_ahora:>7}"
          f"{total_ahora - total_base:>+7}{total_sep:>5}   de {TOTAL_TIRILLAS} tirillas")

    # ---------------- calidad ----------------
    # El conteo de tirillas con datos NO es la metrica final: una grilla mal
    # puesta entrega 37 intensidades igual de plausibles y todas corridas. Lo
    # que cuenta es cuantas tienen la celda con forma de membrana real.
    lo, hi = RATIO_SANO
    con_ratio = [(r["archivo"], t["strip"], t["ratio_celda"])
                 for r in resultados for t in r["tirillas"]
                 if t["con_datos"] and t["ratio_celda"] is not None]
    sanas = [x for x in con_ratio if lo <= x[2] <= hi]
    raros = [x for x in con_ratio if not (lo <= x[2] <= hi)]
    print(f"\nCALIDAD   tirillas SANAS (ratio {lo}-{hi}): {len(sanas)}"
          f"   |   sospechosas: {len(raros)}")
    print(f"SEPARACION  {total_sep}/{TOTAL_TIRILLAS} tirillas separadas "
          f"(cada imagen deberia dar {N_STRIPS}: las 6 membranas tienen 4 fisicas)")
    faltan = [(r["archivo"], r.get("separadas", 0))
              for r in resultados if r.get("separadas", 0) < N_STRIPS]
    if faltan:
        print("  techo estructural - imagenes que no separan las 4:")
        for a, s in faltan:
            print(f"    {a}: {s}/{N_STRIPS}")
    # Firma de grilla: lo que el ratio no ve. Una grilla con forma sana puede
    # estar corrida media fila, y entonces un Reference del par se desploma y
    # el PBS se monta sobre senal real.
    rotas = []
    for r in resultados:
        for t in r["tirillas"]:
            fi = t.get("firma") or {}
            if not t["con_datos"] or not fi:
                continue
            motivos = []
            if fi.get("bal_par") is not None and fi["bal_par"] < BAL_PAR_PISO:
                motivos.append(f"par {fi['peor_par']} desbalanceado "
                               f"(bal={fi['bal_par']} < {BAL_PAR_PISO})")
            if fi.get("colapsados", 0) > 0:
                motivos.append(f"{fi['colapsados']} nodos colapsados")
            if motivos:
                rotas.append((r["archivo"], t["strip"], ", ".join(motivos),
                              r["protegida"]))
    print(f"FIRMA     tirillas con firma de grilla ROTA: {len(rotas)}")
    if rotas:
        print("  (entregan numeros pero la grilla no esta donde debe:)")
        for a, s, m, prot in rotas:
            print(f"    {a} M{s}: {m}" + ("   !! PROTEGIDA !!" if prot else ""))

    if raros:
        print(f"\nATENCION - tirillas aceptadas con relacion de celda fuera de {lo}-{hi}")
        print("(posible grilla mal puesta, revisar la imagen debug):")
        for a, s, ra in raros:
            print(f"  {a} M{s}: ratio={ra}")

    print(f"\nImagenes debug en: {DIR_DEBUG}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"resultados": resultados,
                       "total_separadas": total_sep,
                       "sanas": len(sanas),
                       "sospechosas": len(raros),
                       "firma_rota": [list(x) for x in rotas],
                       "total_base": total_base,
                       "total_ahora": total_ahora,
                       "regresiones": regresiones,
                       "mejoras": mejoras}, fh, indent=2, ensure_ascii=False)
        print(f"JSON: {args.json}")

    if regresiones:
        print("\nRESULTADO: REGRESION - revertir el cambio.")
        return 1
    print("\nRESULTADO: sin regresiones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
