"""
Validación profunda del reporte de Sw620 colon-mango.
Verifica:
  - Consistencia entre tirillas (réplicas)
  - Rango de intensidades plausible
  - Veredictos biológicamente razonables
  - Métricas de calidad del experimento
"""
import sys
import numpy as np
from motor_apoptosis import (
    AnalizadorGrillaARY009,
    promediar_cuantificaciones,
    calcular_fold_change_con_error,
)
from mapa_array import PRO_APOPTOTICAS, ANTI_APOPTOTICAS


def main():
    ruta = 'fotos/Sw620 colon-mango_convertido.png'
    print(f"\n{'=' * 80}")
    print(f"VALIDACIÓN DEL REPORTE — {ruta}")
    print(f"{'=' * 80}\n")

    a = AnalizadorGrillaARY009(radio_muestreo=6)
    intensidades = a.analizar_imagen(ruta, n_strips=4)

    if len(intensidades) < 4:
        print("ERROR: Se necesitan 4 strips")
        return

    # Cuantificar cada strip
    cuant = [a.cuantificar(ints) for ints in intensidades]
    ctrl_avg = promediar_cuantificaciones(cuant[0:2])
    trat_avg = promediar_cuantificaciones(cuant[2:4])
    reporte = calcular_fold_change_con_error(ctrl_avg, trat_avg)

    # ── PARTE 1: Métricas técnicas de la membrana ──
    print("┌" + "─" * 78 + "┐")
    print("│ PARTE 1 — CALIDAD TÉCNICA DE LA MEMBRANA" + " " * 36 + "│")
    print("└" + "─" * 78 + "┘\n")

    print("  Referencias y PBS por tirilla:")
    print(f"  {'Tirilla':<12} {'Ref bruta':>10} {'PBS bruta':>10} {'Ratio R/P':>10} {'Eval':>15}")
    for i, c in enumerate(cuant):
        nombre = ['T1 (Ctrl)', 'T2 (Ctrl)', 'T3 (Trat)', 'T4 (Trat)'][i]
        ref = c.get('referencia', 0)
        pbs = c.get('pbs', 0)
        ratio = ref / max(0.1, pbs)
        if ratio >= 10:
            eval_str = "✓ EXCELENTE"
        elif ratio >= 5:
            eval_str = "✓ Bueno"
        elif ratio >= 2:
            eval_str = "~ Aceptable"
        else:
            eval_str = "✗ MAL"
        print(f"  {nombre:<12} {ref:>10.1f} {pbs:>10.1f} {ratio:>10.2f} {eval_str:>15}")

    # Consistencia de referencias
    refs_all = [c.get('referencia', 0) for c in cuant]
    cv_refs = np.std(refs_all) / np.mean(refs_all) * 100 if np.mean(refs_all) > 0 else 0
    print(f"\n  CV (coef. variación) de las 4 referencias: {cv_refs:.1f}%")
    if cv_refs < 15:
        print(f"  → ✓ Las 4 membranas se revelaron con exposición similar")
    elif cv_refs < 30:
        print(f"  → ~ Variación moderada en exposición")
    else:
        print(f"  → ✗ Variación grande - membranas reveladas de forma desigual")

    # ── PARTE 2: Consistencia entre réplicas ──
    print("\n┌" + "─" * 78 + "┐")
    print("│ PARTE 2 — CONSISTENCIA ENTRE RÉPLICAS (T1≈T2 y T3≈T4?)" + " " * 22 + "│")
    print("└" + "─" * 78 + "┘\n")

    # Para cada proteína, calcular diferencia entre T1 y T2 (control) y T3 y T4 (trat)
    consistencia_ctrl = []
    consistencia_trat = []
    for prot in reporte.keys():
        v1 = cuant[0].get('proteinas', {}).get(prot, {}).get('normalizada', 0)
        v2 = cuant[1].get('proteinas', {}).get(prot, {}).get('normalizada', 0)
        v3 = cuant[2].get('proteinas', {}).get(prot, {}).get('normalizada', 0)
        v4 = cuant[3].get('proteinas', {}).get(prot, {}).get('normalizada', 0)
        mc = (v1 + v2) / 2
        mt = (v3 + v4) / 2
        if mc > 0.005:
            consistencia_ctrl.append(abs(v1 - v2) / mc)
        if mt > 0.005:
            consistencia_trat.append(abs(v3 - v4) / mt)

    cv_ctrl = np.mean(consistencia_ctrl) * 100 if consistencia_ctrl else 0
    cv_trat = np.mean(consistencia_trat) * 100 if consistencia_trat else 0
    print(f"  Variación promedio entre T1 y T2 (Control):     {cv_ctrl:.1f}%")
    print(f"  Variación promedio entre T3 y T4 (Tratamiento): {cv_trat:.1f}%")

    if cv_ctrl < 20 and cv_trat < 20:
        print(f"  → ✓ Réplicas internas MUY consistentes (CV < 20%)")
    elif cv_ctrl < 40 and cv_trat < 40:
        print(f"  → ~ Réplicas con variación aceptable (CV 20-40%)")
    else:
        print(f"  → ✗ Réplicas muy variables - resultado poco confiable")

    # ── PARTE 3: Análisis biológico ──
    print("\n┌" + "─" * 78 + "┐")
    print("│ PARTE 3 — VEREDICTO BIOLÓGICO" + " " * 47 + "│")
    print("└" + "─" * 78 + "┘\n")

    n_pro_up = sum(1 for p, d in reporte.items() if d['tipo'] == 'pro-apoptótica'
                    and 'AUMENTÓ' in d['estado'])
    n_pro_down = sum(1 for p, d in reporte.items() if d['tipo'] == 'pro-apoptótica'
                      and 'DISMINUYÓ' in d['estado'])
    n_anti_up = sum(1 for p, d in reporte.items() if d['tipo'] == 'anti-apoptótica'
                     and 'AUMENTÓ' in d['estado'])
    n_anti_down = sum(1 for p, d in reporte.items() if d['tipo'] == 'anti-apoptótica'
                       and 'DISMINUYÓ' in d['estado'])

    print(f"  Pro-apoptóticas que SUBIERON:  {n_pro_up:>3}  (señal de apoptosis activa)")
    print(f"  Pro-apoptóticas que BAJARON:   {n_pro_down:>3}")
    print(f"  Anti-apoptóticas que SUBIERON: {n_anti_up:>3}  (señal de resistencia)")
    print(f"  Anti-apoptóticas que BAJARON:  {n_anti_down:>3}  (señal de muerte celular)")

    score = (n_pro_up + n_anti_down) - (n_pro_down + n_anti_up)
    print(f"\n  Puntaje apoptótico: {score:+d}")
    if score >= 5:
        print(f"  → ✓ APOPTOSIS ACTIVA (tratamiento funcionando)")
    elif score >= 2:
        print(f"  → ↗ Tendencia pro-apoptótica leve")
    elif score <= -5:
        print(f"  → ✗ RESISTENCIA (no hay muerte celular)")
    elif score <= -2:
        print(f"  → ↘ Tendencia anti-apoptótica")
    else:
        print(f"  → ─ Sin cambio significativo")

    # ── PARTE 4: Top hallazgos ──
    print("\n┌" + "─" * 78 + "┐")
    print("│ PARTE 4 — TOP 5 PROTEÍNAS CON MAYOR CAMBIO" + " " * 35 + "│")
    print("└" + "─" * 78 + "┘\n")

    cambios = []
    for prot, d in reporte.items():
        fc = d.get('fold_change', 1.0)
        if fc not in (float('inf'), 0.0) and fc > 0:
            magnitud = max(fc, 1 / fc)
            cambios.append((prot, fc, d['estado'], magnitud, d.get('tipo', 'otra')))
    cambios.sort(key=lambda x: -x[3])

    print(f"  {'Proteína':<25} {'Fold':>8} {'Tipo':<22} {'Veredicto'}")
    print(f"  {'-' * 78}")
    for prot, fc, estado, _, tipo in cambios[:10]:
        print(f"  {prot:<25} {fc:>8.3f} {tipo:<22} {estado}")

    # ── CONCLUSIÓN ──
    print("\n┌" + "─" * 78 + "┐")
    print("│ CONCLUSIÓN GENERAL DEL EXPERIMENTO" + " " * 43 + "│")
    print("└" + "─" * 78 + "┘\n")

    confiabilidad = "ALTA"
    issues = []
    if min(refs_all) < 30:
        confiabilidad = "MEDIA"; issues.append("alguna tirilla con referencia muy baja")
    if cv_refs > 30:
        confiabilidad = "MEDIA"; issues.append("variación grande entre membranas")
    if cv_ctrl > 40 or cv_trat > 40:
        confiabilidad = "BAJA"; issues.append("réplicas internas muy variables")

    print(f"  Confiabilidad de los datos: {confiabilidad}")
    if issues:
        print(f"  Problemas detectados: {', '.join(issues)}")
    else:
        print(f"  → Los datos son confiables para sacar conclusiones biológicas")

    print(f"  Proteínas analizadas: {len(reporte)}")
    print(f"  Cambios significativos detectados: {n_pro_up + n_pro_down + n_anti_up + n_anti_down}")
    print()


if __name__ == "__main__":
    main()
