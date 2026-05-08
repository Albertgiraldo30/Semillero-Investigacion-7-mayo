# 🔬 Analizador Proteome Profiler ARY009

**Semillero de Investigación Biomédica — Detección de Cáncer**  
Instituto Tecnológico Metropolitano (ITM)

## Descripción

Pipeline automatizado para el análisis de membranas del **Proteome Profiler Human Apoptosis Array Kit (ARY009)** de R&D Systems. El sistema detecta, cuantifica y compara la expresión de **35 proteínas relacionadas con apoptosis** entre condiciones de control y tratamiento.

## ¿Qué hace?

1. **Carga** una imagen escaneada con 4 tirillas de membrana
2. **Separa** las tirillas automáticamente usando Black-Hat morfológico
3. **Detecta** los Reference Spots para anclar una grilla geométrica (5 columnas × 24 filas)
4. **Muestrea** la intensidad en cada posición de proteína con Safe Snapping iterativo + ICP
5. **Cuantifica**: resta fondo PBS, normaliza por Reference Spots, promedia réplicas
6. **Calcula** Fold Change con propagación de error entre Control y Tratamiento
7. **Interpreta** los resultados con un puntaje apoptótico y explicaciones en español
8. **Exporta** a Excel con formato profesional

## Captura

La interfaz incluye 4 pestañas: Reporte numérico, Interpretación biológica, Gráficos estilo paper, y Mapa Visual de la detección.

## Requisitos

```bash
pip install opencv-python numpy matplotlib customtkinter Pillow openpyxl
```

**Opcional** (para archivos WSI de escáner):
```bash
pip install openslide-python openslide-bin
```

## Uso

```bash
python interfaz.py
```

1. Click en **"Cargar imagen del escáner"** y selecciona la imagen PNG/TIFF
2. Click en **"Analizar Membranas"**
3. Revisa los resultados en las 4 pestañas
4. Si la detección automática falla en alguna tirilla, usa **"Ajustar referencias manualmente"**
5. Exporta a Excel con el botón inferior

## Estructura del Proyecto

| Archivo | Descripción |
|---------|-------------|
| `interfaz.py` | GUI principal (CustomTkinter + Matplotlib) |
| `motor_apoptosis.py` | Motor de análisis: detección, grilla, cuantificación |
| `mapa_array.py` | Mapa oficial del kit ARY009 (proteínas → coordenadas) |
| `utils_wsi.py` | Utilidades para Whole Slide Images |
| `etl_scn_converter.py` | Conversor masivo .SCN → TIFF/PNG |
| `generar_mapa_ary009.py` | Genera la imagen de referencia del mapa |

## Esquema del Kit ARY009

- **Tirillas 1-2**: Control (2 réplicas internas)
- **Tirillas 3-4**: Tratamiento (2 réplicas internas)
- **35 proteínas** clasificadas como pro-apoptóticas o anti-apoptóticas
- **Reference Spots** en 3 esquinas para normalización
- **PBS** como control negativo (fondo)

## Autores

Semillero de Investigación Biomédica — ITM, 2026
