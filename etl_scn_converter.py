"""
etl_scn_converter.py — Pipeline ETL para conversión masiva de archivos Leica .SCN a TIFF/PNG.

Convierte Whole Slide Images (WSI) en formato .scn a tiles de alta resolución
(nivel 0) en formato TIFF o PNG, filtrando automáticamente el fondo blanco
para exportar únicamente regiones con tejido.

Uso:
    # Convertir todos los .scn de un directorio a tiles TIFF de 512x512:
    python etl_scn_converter.py --input ./datos_scn --output ./tiles --formato tif --tile-size 512

    # Generar solo thumbnails de baja resolución:
    python etl_scn_converter.py --input ./datos_scn --output ./previews --thumbnail-only

    # Extraer un ROI específico de todos los archivos:
    python etl_scn_converter.py --input ./datos_scn --output ./rois --roi 1000,2000,1024,1024

    # Procesamiento paralelo con 4 workers:
    python etl_scn_converter.py --input ./datos_scn --output ./tiles --workers 4

Dependencias:
    pip install openslide-python openslide-bin numpy opencv-python Pillow tqdm

Autor: Semillero Biomédica ITM — Detección de Cáncer
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# Dependencia opcional para barra de progreso
try:
    from tqdm import tqdm
    TQDM_DISPONIBLE = True
except ImportError:
    TQDM_DISPONIBLE = False

# Importar motor de lectura WSI
from motor_apoptosis import LectorWSI
from utils_wsi import filtrar_fondo

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s [%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('ETL_SCN')

# Extensiones WSI soportadas por este conversor
EXTENSIONES_WSI = {'.scn', '.svs', '.ndpi', '.mrxs', '.vms', '.bif'}


# =============================================================================
# Funciones de conversión
# =============================================================================

def convertir_slide_a_tiles(
    ruta_scn: str,
    dir_salida: str,
    tile_size: int = 512,
    nivel: int = 0,
    formato: str = 'tif',
    solo_tejido: bool = True,
    umbral_blancura: float = 0.85,
    calidad_jpeg: int = 95,
    overlap: int = 0,
) -> dict:
    """
    Convierte un archivo WSI en tiles individuales de alta resolución.

    Cada tile se guarda como un archivo separado con nombre que incluye
    las coordenadas de origen, permitiendo reconstrucción futura.

    Args:
        ruta_scn: Ruta al archivo WSI (.scn).
        dir_salida: Directorio donde se guardarán los tiles.
        tile_size: Tamaño del tile en píxeles (512 o 1024 recomendado).
        nivel: Nivel piramidal a extraer (0 = máxima resolución).
        formato: Formato de salida ('tif', 'png', 'jpg').
        solo_tejido: Si True, descarta tiles que son fondo blanco.
        umbral_blancura: Fracción de blancura para filtrar fondo.
        calidad_jpeg: Calidad JPEG si formato='jpg' (1-100).
        overlap: Píxeles de overlap entre tiles adyacentes.

    Returns:
        Diccionario con estadísticas de la conversión:
            - archivo_origen
            - tiles_exportados
            - tiles_descartados (fondo)
            - tiempo_segundos
            - directorio_salida
    """
    nombre_base = Path(ruta_scn).stem
    dir_slide = os.path.join(dir_salida, nombre_base)
    os.makedirs(dir_slide, exist_ok=True)

    inicio = time.time()
    tiles_exportados = 0
    tiles_descartados = 0

    ext_salida = f'.{formato.lower().strip(".")}'

    # Parámetros de compresión según formato
    params_escritura = []
    if ext_salida == '.jpg':
        params_escritura = [cv2.IMWRITE_JPEG_QUALITY, calidad_jpeg]
    elif ext_salida == '.png':
        params_escritura = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # Compresión rápida

    with LectorWSI(ruta_scn) as lector:
        meta = lector.obtener_metadata()
        logger.info(
            f"Procesando: {nombre_base} — "
            f"Dimensiones nivel 0: {meta['dimensiones_nivel_0']} — "
            f"Niveles: {meta['niveles']}"
        )

        # Guardar metadatos del slide
        _guardar_metadata(dir_slide, meta, ruta_scn)

        # Iterar tiles con generador (memoria constante)
        for x, y, tile_np in lector.iterar_tiles(
            tile_size=tile_size,
            nivel=nivel,
            solo_tejido=solo_tejido,
            umbral_blancura=umbral_blancura,
            overlap=overlap,
        ):
            # Nombre del archivo: slide_x{coord}_y{coord}_s{size}.ext
            nombre_tile = f"{nombre_base}_x{x}_y{y}_s{tile_size}{ext_salida}"
            ruta_tile = os.path.join(dir_slide, nombre_tile)

            # OpenSlide produce RGB; OpenCV espera BGR para escritura
            tile_bgr = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta_tile, tile_bgr, params_escritura)
            tiles_exportados += 1

        # Calcular tiles descartados
        dim_nivel = meta['dimensiones_por_nivel'][nivel]
        total_posible = (
            ((dim_nivel[0] + tile_size - 1) // tile_size) *
            ((dim_nivel[1] + tile_size - 1) // tile_size)
        )
        tiles_descartados = total_posible - tiles_exportados

    tiempo_total = round(time.time() - inicio, 2)

    resultado = {
        'archivo_origen': nombre_base,
        'tiles_exportados': tiles_exportados,
        'tiles_descartados': tiles_descartados,
        'tiempo_segundos': tiempo_total,
        'directorio_salida': dir_slide,
    }

    logger.info(
        f"✓ {nombre_base}: {tiles_exportados} tiles exportados, "
        f"{tiles_descartados} descartados (fondo) — {tiempo_total}s"
    )

    return resultado


def generar_thumbnail(
    ruta_scn: str,
    dir_salida: str,
    max_dim: int = 2048,
    formato: str = 'png'
) -> str:
    """
    Genera un thumbnail (vista previa) de un archivo WSI.

    Args:
        ruta_scn: Ruta al archivo WSI.
        dir_salida: Directorio de salida.
        max_dim: Dimensión máxima del thumbnail.
        formato: Formato de salida.

    Returns:
        Ruta al archivo del thumbnail generado.
    """
    nombre_base = Path(ruta_scn).stem
    os.makedirs(dir_salida, exist_ok=True)

    with LectorWSI(ruta_scn) as lector:
        thumb = lector.obtener_thumbnail(max_dim=max_dim)
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)

        ruta_salida = os.path.join(dir_salida, f"{nombre_base}_thumbnail.{formato}")
        cv2.imwrite(ruta_salida, thumb_bgr)

    logger.info(f"✓ Thumbnail generado: {ruta_salida} ({thumb.shape[1]}x{thumb.shape[0]})")
    return ruta_salida


def extraer_roi_especifico(
    ruta_scn: str,
    dir_salida: str,
    roi: Tuple[int, int, int, int],
    nivel: int = 0,
    formato: str = 'tif'
) -> str:
    """
    Extrae un ROI específico de un archivo WSI.

    Args:
        ruta_scn: Ruta al archivo WSI.
        dir_salida: Directorio de salida.
        roi: Tupla (x, y, ancho, alto) en coordenadas del nivel 0.
        nivel: Nivel piramidal a leer.
        formato: Formato de salida.

    Returns:
        Ruta al archivo del ROI extraído.
    """
    nombre_base = Path(ruta_scn).stem
    os.makedirs(dir_salida, exist_ok=True)
    x, y, ancho, alto = roi

    with LectorWSI(ruta_scn) as lector:
        roi_np = lector.leer_roi(x, y, ancho, alto, nivel)
        roi_bgr = cv2.cvtColor(roi_np, cv2.COLOR_RGB2BGR)

        nombre_roi = f"{nombre_base}_roi_x{x}_y{y}_w{ancho}_h{alto}.{formato}"
        ruta_salida = os.path.join(dir_salida, nombre_roi)
        cv2.imwrite(ruta_salida, roi_bgr)

    logger.info(f"✓ ROI extraído: {ruta_salida} ({ancho}x{alto}px)")
    return ruta_salida


# =============================================================================
# Funciones auxiliares
# =============================================================================

def _guardar_metadata(dir_slide: str, meta: dict, ruta_origen: str):
    """Guarda los metadatos del slide en un archivo JSON."""
    import json
    meta_serializable = {}
    for k, v in meta.items():
        if isinstance(v, tuple):
            meta_serializable[k] = list(v)
        elif isinstance(v, list) and v and isinstance(v[0], tuple):
            meta_serializable[k] = [list(t) for t in v]
        else:
            meta_serializable[k] = v

    meta_serializable['archivo_origen'] = str(ruta_origen)

    ruta_meta = os.path.join(dir_slide, '_metadata.json')
    with open(ruta_meta, 'w', encoding='utf-8') as f:
        json.dump(meta_serializable, f, indent=2, ensure_ascii=False)


def buscar_archivos_wsi(directorio: str) -> list:
    """
    Busca recursivamente archivos WSI en un directorio.

    Args:
        directorio: Ruta al directorio raíz de búsqueda.

    Returns:
        Lista de rutas absolutas a archivos WSI encontrados.
    """
    archivos = []
    for root, dirs, files in os.walk(directorio):
        for f in files:
            if Path(f).suffix.lower() in EXTENSIONES_WSI:
                archivos.append(os.path.join(root, f))

    archivos.sort()
    return archivos


# =============================================================================
# Pipeline principal
# =============================================================================

def ejecutar_pipeline(args: argparse.Namespace):
    """
    Ejecuta el pipeline ETL completo según los argumentos de la CLI.

    Soporta tres modos de operación:
    1. --thumbnail-only: Solo genera vistas previas de baja resolución.
    2. --roi x,y,w,h: Extrae un ROI específico de cada archivo.
    3. (default): Convierte cada WSI en tiles de resolución máxima.

    Args:
        args: Argumentos parseados de la línea de comandos.
    """
    logger.info("=" * 60)
    logger.info("  ETL SCN Converter — Semillero Biomédica ITM")
    logger.info("=" * 60)

    # Buscar archivos WSI
    archivos = buscar_archivos_wsi(args.input)

    if not archivos:
        logger.error(f"No se encontraron archivos WSI en: {args.input}")
        sys.exit(1)

    logger.info(f"Encontrados {len(archivos)} archivos WSI:")
    for a in archivos:
        logger.info(f"  → {os.path.basename(a)}")

    os.makedirs(args.output, exist_ok=True)

    resultados = []
    inicio_global = time.time()

    # ---- MODO 1: Solo thumbnails ----
    if args.thumbnail_only:
        logger.info(f"\nModo: Generación de thumbnails (max {args.thumb_size}px)")
        for ruta in archivos:
            try:
                generar_thumbnail(ruta, args.output, args.thumb_size, args.formato)
            except Exception as e:
                logger.error(f"Error procesando {os.path.basename(ruta)}: {e}")

    # ---- MODO 2: ROI específico ----
    elif args.roi:
        roi_coords = tuple(map(int, args.roi.split(',')))
        if len(roi_coords) != 4:
            logger.error("El ROI debe tener exactamente 4 valores: x,y,ancho,alto")
            sys.exit(1)

        logger.info(f"\nModo: Extracción de ROI {roi_coords}")
        for ruta in archivos:
            try:
                extraer_roi_especifico(ruta, args.output, roi_coords, args.nivel, args.formato)
            except Exception as e:
                logger.error(f"Error procesando {os.path.basename(ruta)}: {e}")

    # ---- MODO 3: Conversión completa a tiles ----
    else:
        logger.info(
            f"\nModo: Conversión a tiles — "
            f"tile_size={args.tile_size}px, nivel={args.nivel}, "
            f"formato={args.formato}, workers={args.workers}"
        )

        if args.workers > 1:
            # Procesamiento paralelo
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futuros = {
                    executor.submit(
                        convertir_slide_a_tiles,
                        ruta, args.output, args.tile_size, args.nivel,
                        args.formato, not args.incluir_fondo,
                        args.umbral_blancura, overlap=args.overlap
                    ): ruta
                    for ruta in archivos
                }

                for futuro in as_completed(futuros):
                    ruta = futuros[futuro]
                    try:
                        resultado = futuro.result()
                        resultados.append(resultado)
                    except Exception as e:
                        logger.error(f"Error procesando {os.path.basename(ruta)}: {e}")
        else:
            # Procesamiento secuencial
            iterador = archivos
            if TQDM_DISPONIBLE:
                iterador = tqdm(archivos, desc="Convirtiendo slides", unit="slide")

            for ruta in iterador:
                try:
                    resultado = convertir_slide_a_tiles(
                        ruta_scn=ruta,
                        dir_salida=args.output,
                        tile_size=args.tile_size,
                        nivel=args.nivel,
                        formato=args.formato,
                        solo_tejido=not args.incluir_fondo,
                        umbral_blancura=args.umbral_blancura,
                        overlap=args.overlap,
                    )
                    resultados.append(resultado)
                except Exception as e:
                    logger.error(f"Error procesando {os.path.basename(ruta)}: {e}")

    # ---- Resumen final ----
    tiempo_global = round(time.time() - inicio_global, 2)

    logger.info("\n" + "=" * 60)
    logger.info("  RESUMEN DE CONVERSIÓN")
    logger.info("=" * 60)

    total_tiles = sum(r.get('tiles_exportados', 0) for r in resultados)
    total_descartados = sum(r.get('tiles_descartados', 0) for r in resultados)

    if resultados:
        for r in resultados:
            logger.info(
                f"  {r['archivo_origen']}: "
                f"{r['tiles_exportados']} tiles ({r['tiempo_segundos']}s)"
            )

    logger.info(f"\n  Total tiles exportados: {total_tiles}")
    logger.info(f"  Total tiles descartados (fondo): {total_descartados}")
    logger.info(f"  Tiempo total: {tiempo_global}s")
    logger.info(f"  Directorio de salida: {os.path.abspath(args.output)}")
    logger.info("=" * 60)


# =============================================================================
# CLI (Command Line Interface)
# =============================================================================

def crear_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "ETL SCN Converter — Convierte archivos Leica .SCN (WSI) a tiles "
            "TIFF/PNG de alta resolución para análisis de apoptosis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Conversión estándar a tiles TIFF de 512x512:
  python etl_scn_converter.py --input ./datos --output ./tiles

  # Tiles de 1024x1024 en formato PNG:
  python etl_scn_converter.py --input ./datos --output ./tiles --tile-size 1024 --formato png

  # Solo thumbnails de previsualización:
  python etl_scn_converter.py --input ./datos --output ./previews --thumbnail-only

  # Extraer ROI específico (x=1000, y=2000, ancho=1024, alto=1024):
  python etl_scn_converter.py --input ./datos --output ./rois --roi 1000,2000,1024,1024

  # Procesamiento paralelo con 4 hilos:
  python etl_scn_converter.py --input ./datos --output ./tiles --workers 4
        """
    )

    # Argumentos obligatorios
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Directorio de entrada con archivos WSI (.scn, .svs, etc.)'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Directorio de salida para tiles/thumbnails generados'
    )

    # Configuración de tiles
    parser.add_argument(
        '--tile-size', '-s',
        type=int,
        default=512,
        choices=[256, 512, 1024, 2048],
        help='Tamaño del tile en píxeles (default: 512)'
    )
    parser.add_argument(
        '--nivel', '-n',
        type=int,
        default=0,
        help='Nivel piramidal a extraer (0=máx resolución, default: 0)'
    )
    parser.add_argument(
        '--formato', '-f',
        default='tif',
        choices=['tif', 'png', 'jpg'],
        help='Formato de salida (default: tif)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=0,
        help='Píxeles de overlap entre tiles adyacentes (default: 0)'
    )

    # Filtrado
    parser.add_argument(
        '--incluir-fondo',
        action='store_true',
        default=False,
        help='Incluir tiles de fondo blanco (por defecto se filtran)'
    )
    parser.add_argument(
        '--umbral-blancura',
        type=float,
        default=0.85,
        help='Fracción de blancura para descartar fondo (default: 0.85)'
    )

    # Modos alternativos
    parser.add_argument(
        '--thumbnail-only',
        action='store_true',
        help='Solo generar thumbnails de baja resolución'
    )
    parser.add_argument(
        '--thumb-size',
        type=int,
        default=2048,
        help='Dimensión máxima del thumbnail (default: 2048)'
    )
    parser.add_argument(
        '--roi',
        type=str,
        default=None,
        help='ROI específico a extraer: x,y,ancho,alto (coordenadas nivel 0)'
    )

    # Rendimiento
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Número de hilos para procesamiento paralelo (default: 1)'
    )

    return parser


if __name__ == '__main__':
    parser = crear_parser()
    args = parser.parse_args()
    ejecutar_pipeline(args)
