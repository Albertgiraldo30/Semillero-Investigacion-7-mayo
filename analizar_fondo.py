"""
Analiza el fondo de tu imagen ARY009 para detectar problemas:
- Fondo no uniforme entre tirillas
- Fondo no uniforme dentro de una tirilla
- Valores fuera de rango razonable

Uso:  python analizar_fondo.py "ruta/a/tu/imagen.png"
"""
import sys
import os
import numpy as np
import cv2

def analizar(ruta):
    if not os.path.exists(ruta):
        print(f"ERROR: no existe {ruta}")
        return

    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: no se pudo abrir {ruta}")
        return

    if img.ndim == 3:
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gris = img

    h, w = gris.shape
    profundidad = 16 if gris.dtype == np.uint16 else 8
    max_val = 65535 if profundidad == 16 else 255

    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE FONDO: {os.path.basename(ruta)}")
    print(f"{'='*70}\n")
    print(f"Dimensiones: {w} × {h} px")
    print(f"Profundidad: {profundidad}-bit (rango {0}-{max_val})\n")

    # Estadísticas globales
    print(f"{'─'*70}")
    print("ESTADÍSTICAS GLOBALES DE LA IMAGEN")
    print(f"{'─'*70}")
    print(f"Min:     {gris.min():>6}  (más oscuro = spot más fuerte)")
    print(f"Max:     {gris.max():>6}  (más claro = fondo)")
    print(f"Mediana: {int(np.median(gris)):>6}  (color típico de la imagen)")
    print(f"Media:   {gris.mean():>6.1f}\n")

    # Detectar el fondo (los píxeles MÁS CLAROS)
    fondo_aprox = int(np.percentile(gris, 95))  # 95th percentile = fondo
    print(f"FONDO APROXIMADO (95% más claro): {fondo_aprox}/{max_val}  "
          f"({100*fondo_aprox/max_val:.1f}% del máximo)")

    if profundidad == 8:
        if fondo_aprox >= 245:
            print(f"  ✓ EXCELENTE: fondo casi blanco puro")
        elif fondo_aprox >= 230:
            print(f"  ✓ BIEN: fondo claro (variación normal de escáner)")
        elif fondo_aprox >= 200:
            print(f"  ~ ACEPTABLE: fondo levemente gris (verificar exposición)")
        else:
            print(f"  ✗ PROBLEMA: fondo muy oscuro (reduce sensibilidad)")

    # Dividir en 4 tirillas verticales y analizar cada una
    print(f"\n{'─'*70}")
    print("ANÁLISIS POR TIRILLA (división vertical en 4)")
    print(f"{'─'*70}")
    print(f"{'Tirilla':<12} {'fondo':<8} {'min':<6} {'rango_usado':<14} {'fondo_uniforme'}")
    ancho_tirilla = w // 4
    fondos = []
    for i in range(4):
        x0 = i * ancho_tirilla
        x1 = (i + 1) * ancho_tirilla
        sub = gris[:, x0:x1]
        f95 = int(np.percentile(sub, 95))
        mn = sub.min()
        rango = f95 - mn
        fondos.append(f95)

        # Verificar uniformidad: dividir tirilla en mitad superior e inferior
        sub_sup = sub[:h//2]
        sub_inf = sub[h//2:]
        f_sup = int(np.percentile(sub_sup, 95))
        f_inf = int(np.percentile(sub_inf, 95))
        delta = abs(f_sup - f_inf)
        if delta < 10:
            unif = "✓ uniforme"
        elif delta < 25:
            unif = f"~ leve gradiente (Δ={delta})"
        else:
            unif = f"✗ DESIGUAL (Δ={delta})"

        nombre = ['Ctrl-1', 'Ctrl-2', 'Trat-1', 'Trat-2'][i]
        print(f"  {nombre:<10} {f95:<8} {mn:<6} {rango:<14} {unif}")

    print(f"\n{'─'*70}")
    print("CONSISTENCIA DEL FONDO ENTRE TIRILLAS")
    print(f"{'─'*70}")
    fondo_min = min(fondos)
    fondo_max = max(fondos)
    delta_max = fondo_max - fondo_min
    if delta_max < 5:
        print(f"  ✓ EXCELENTE: las 4 tirillas tienen fondo casi idéntico")
    elif delta_max < 15:
        print(f"  ✓ ACEPTABLE: variación leve entre tirillas (Δ={delta_max})")
    elif delta_max < 30:
        print(f"  ~ ATENCIÓN: variación moderada (Δ={delta_max}) — afectará comparación Ctrl vs Trat")
    else:
        print(f"  ✗ PROBLEMA: variación grande (Δ={delta_max}) — los resultados pueden ser artefacto")

    print(f"\n{'─'*70}")
    print("CONCLUSIÓN")
    print(f"{'─'*70}")
    if delta_max < 15 and fondo_aprox >= 230:
        print("  ✓ Tu imagen tiene buena calidad técnica. Los resultados deberían")
        print("    ser confiables. Cualquier diferencia entre Ctrl y Trat es real.")
    elif delta_max < 30:
        print("  ~ La imagen tiene variación moderada en el fondo. Algunos")
        print("    resultados pueden estar levemente sesgados (~10-20% de error).")
        print("    Considere re-escanear con configuración uniforme.")
    else:
        print("  ✗ La imagen tiene problemas significativos de fondo.")
        print("    Muchos de los resultados pueden ser ARTEFACTO del escaneo.")
        print("    Re-escanee usando MISMA exposición/calibración para las 4 tirillas.")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analizar_fondo.py <ruta_imagen.png>")
        print('Ejemplo: python analizar_fondo.py "fotos/Sw620 colon-mango.png"')
        sys.exit(1)
    analizar(sys.argv[1])
