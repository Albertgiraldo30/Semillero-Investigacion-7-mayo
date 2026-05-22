"""Comparar reportes de las dos versiones de la imagen Sw620."""
import os
from motor_apoptosis import (
    AnalizadorGrillaARY009,
    promediar_cuantificaciones,
    calcular_fold_change_con_error,
)


def procesar(ruta):
    """Procesa imagen y retorna reporte completo."""
    a = AnalizadorGrillaARY009(radio_muestreo=6)
    intensidades = a.analizar_imagen(ruta, n_strips=4)
    cuant_lista = [a.cuantificar(ints) for ints in intensidades]
    ctrl_avg = promediar_cuantificaciones(cuant_lista[0:2])
    trat_avg = promediar_cuantificaciones(cuant_lista[2:4])
    reporte = calcular_fold_change_con_error(ctrl_avg, trat_avg)
    return {
        'cuant_lista': cuant_lista,
        'ctrl_avg': ctrl_avg,
        'trat_avg': trat_avg,
        'reporte': reporte,
    }


def main():
    rutas = [
        ('PNG original',  'C:/Users/user/Downloads/Sw620 colon-mango.png'),
        ('Convertido',    'fotos/Sw620 colon-mango_convertido.png'),
    ]

    resultados = {}
    for nombre, ruta in rutas:
        if not os.path.exists(ruta):
            print(f"[!] No existe: {ruta}")
            continue
        print(f"\n=== Procesando: {nombre} ===")
        resultados[nombre] = procesar(ruta)

    if len(resultados) < 2:
        print("Se necesitan ambos archivos para comparar.")
        return

    # ── COMPARACIÓN ──
    print("\n" + "=" * 90)
    print("COMPARACIÓN DE REFERENCIAS Y PBS POR TIRILLA")
    print("=" * 90)
    for i in range(4):
        nombre_strip = ['T1 Ctrl', 'T2 Ctrl', 'T3 Trat', 'T4 Trat'][i]
        print(f"\n{nombre_strip}:")
        for nombre, r in resultados.items():
            c = r['cuant_lista'][i]
            ref = c.get('referencia', 0)
            pbs = c.get('pbs', 0)
            ratio = ref / max(0.1, pbs)
            print(f"  {nombre:<15}: Ref={ref:>6.1f}  PBS={pbs:>5.1f}  Ratio={ratio:>5.1f}")

    print("\n" + "=" * 90)
    print("COMPARACIÓN DE VALORES BRUTOS — top 10 proteínas más expresadas (Control)")
    print("=" * 90)

    # Obtener proteínas comunes
    rep1 = resultados['PNG original']['reporte']
    rep2 = resultados['Convertido']['reporte']
    comunes = sorted(set(rep1.keys()) & set(rep2.keys()))

    # Ordenar por intensidad bruta promedio control
    def avg_bruta(rep, prot):
        return rep.get(prot, {}).get('ctrl_bruta', 0)

    top10 = sorted(comunes,
                    key=lambda p: -(avg_bruta(rep1, p) + avg_bruta(rep2, p)) / 2)[:15]

    print(f"\n  {'Proteína':<26} {'PNG Orig (bruta)':<20} {'Convertido (bruta)':<22} {'Diferencia'}")
    print("  " + "-" * 88)
    for prot in top10:
        d1 = rep1.get(prot, {})
        d2 = rep2.get(prot, {})
        c1 = d1.get('ctrl_bruta', 0)
        t1 = d1.get('trat_bruta', 0)
        c2 = d2.get('ctrl_bruta', 0)
        t2 = d2.get('trat_bruta', 0)
        prom1 = (c1 + t1) / 2
        prom2 = (c2 + t2) / 2
        if prom1 > 0.1:
            dif_pct = abs(prom2 - prom1) / prom1 * 100
        else:
            dif_pct = 0
        print(f"  {prot:<26} c={c1:>5.1f} t={t1:>5.1f}      c={c2:>5.1f} t={t2:>5.1f}       {dif_pct:>5.1f}%")

    print("\n" + "=" * 90)
    print("COMPARACIÓN DE VEREDICTOS (¿qué pasó?)")
    print("=" * 90)
    print(f"\n  {'Proteína':<26} {'PNG Original':<25} {'Convertido':<25}")
    print("  " + "-" * 78)

    diferencias = 0
    for prot in sorted(comunes):
        e1 = rep1.get(prot, {}).get('estado', '?')
        e2 = rep2.get(prot, {}).get('estado', '?')
        if e1 == e2:
            indicador = "✓"
        else:
            indicador = "DIFIEREN"
            diferencias += 1
        print(f"  {prot:<26} {e1:<25} {e2:<25}  {indicador}")

    print(f"\n  Resumen: {len(comunes) - diferencias}/{len(comunes)} proteínas con MISMO veredicto.")
    print(f"  Diferencias: {diferencias}/{len(comunes)} ({100 * diferencias / len(comunes):.0f}%)")


if __name__ == "__main__":
    main()
