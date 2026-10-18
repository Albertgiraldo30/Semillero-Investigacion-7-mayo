# -*- coding: utf-8 -*-
"""
Métricas objetivas de centrado de la grilla (verificación externa, 17-oct-2026).
Uso (desde la raíz del proyecto):  python tests/metricas_centrado.py

  tirillas_con_datos     : debe seguir en 23/24 (no empeorar).
  superpuestos           : pares de cuadros a < 1.2r (dos proteínas sobre el mismo spot). Menor = mejor.
  sobre_vecina           : cuadros a > 47% del paso de su posición geométrica (midiendo a la proteína vecina). Menor = mejor.
  dif_duplicados         : mediana de la diferencia relativa entre los 2 duplicados de cada proteína
                           con señal. Si los cuadros están bien puestos, los duplicados se parecen: menor = mejor.
  descentrado_con_senal  : % de cuadros CON mancha cuyo centroide de intensidad queda a > 0.35r del centro.
"""
import os, sys, glob, logging
import numpy as np, cv2
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
logging.basicConfig(level=logging.CRITICAL)
from motor_apoptosis import AnalizadorGrillaARY009, MAPA_PROTEINAS
from mapa_array import T_COLS

def centroide_offset(g, x, y, r):
    h, w = g.shape; R = max(2, int(round(r * 1.0)))
    x0, x1, y0, y1 = max(0, x-R), min(w, x+R+1), max(0, y-R), min(h, y+R+1)
    roi = 255 - g[y0:y1, x0:x1].astype(float); roi -= np.percentile(roi, 40); roi[roi < 0] = 0
    if roi.max() < 12: return None
    roi[roi < 0.5*roi.max()] = 0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (xs*roi).sum()/roi.sum() - x, (ys*roi).sum()/roi.sum() - y

def main():
    tot = dict(tirillas=0, superpuestos=0, sobre_vecina=0, cuadros=0)
    difs, desc = [], []
    for f in sorted(glob.glob(os.path.join(RAIZ, "fotos", "*.scn"))):
        a = AnalizadorGrillaARY009(radio_muestreo=8)
        res = a.analizar_imagen(f, 4)
        tot['tirillas'] += sum(1 for x in res if x)
        fila = []
        for sv in a._strips_visual:
            pos, r, refs = sv.get('posiciones_cache'), sv.get('r_cache'), sv.get('refs')
            if not pos or not refs: fila.append('--'); continue
            tl, tr, bl = refs[:3]
            P = np.array(list(pos.values()), float)
            sup = sum(1 for i in range(len(P)) for j in range(i+1, len(P)) if np.hypot(*(P[i]-P[j])) < 1.2*r)
            colsp = abs(tr[0]-tl[0])*0.21; rowsp = abs(bl[1]-tl[1])/22; vec = 0
            inv = (65535 - sv['strip_orig'].astype(float))/256 if sv['strip_orig'].dtype == np.uint16 else 255.0 - sv['strip_orig']
            bg = np.median(inv); acc = {}
            g = cv2.GaussianBlur(sv['strip_pos'], (3, 3), 0)
            for (c, fi), (px, py) in pos.items():
                t = (fi-1.5)/22; gx = tl[0]+t*(bl[0]-tl[0])+T_COLS[c]*(tr[0]-tl[0]); gy = tl[1]+t*(bl[1]-tl[1])
                if c in 'BCD' and (abs(px-gx) > 0.47*colsp or abs(py-gy) > 0.47*rowsp): vec += 1
                p = MAPA_PROTEINAS[(c, fi)]
                if p not in ('Reference', 'PBS'):
                    acc.setdefault(p, []).append(inv[max(0,py-r):py+r, max(0,px-r):px+r].mean() - bg)
                o = centroide_offset(g, int(px), int(py), r)
                if o is not None: desc.append(np.hypot(*o)/r)
            for v in acc.values():
                if len(v) == 2 and max(v) > 8: difs.append(abs(v[0]-v[1])/((v[0]+v[1])/2)*100)
            tot['superpuestos'] += sup; tot['sobre_vecina'] += vec; tot['cuadros'] += 72
            fila.append(f"sup={sup} vec={vec}")
        print(f"{os.path.basename(f)[:-4]:26s} {fila}")
    print(f"\nTIRILLAS CON DATOS     : {tot['tirillas']}/24")
    print(f"SUPERPUESTOS (pares)   : {tot['superpuestos']}")
    print(f"SOBRE LA VECINA        : {tot['sobre_vecina']} de {tot['cuadros']}")
    print(f"DIF. ENTRE DUPLICADOS  : {np.median(difs):.1f}% (n={len(difs)})")
    print(f"DESCENTRADO CON SEÑAL  : {100*np.mean(np.array(desc) > 0.35):.0f}% de {len(desc)} cuadros con mancha")

if __name__ == "__main__":
    main()
