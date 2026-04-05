"""
utils_wsi.py — Utilidades compartidas para manejo de Whole Slide Images (WSI).

Este módulo proporciona funciones auxiliares reutilizables para:
- Detección de formato WSI vs. formato estándar de imagen.
- Conversión eficiente RGBA → RGB (OpenSlide retorna RGBA).
- Cálculo de grids de tiles con overlap opcional.
- Filtrado de fondo blanco para identificar tejido en tiles.

Autor: Semillero Biomédica ITM — Detección de Cáncer
"""

import os
import logging
import numpy as np
from typing import Tuple, Generator, List

logger = logging.getLogger('utils_wsi')

# Extensiones de archivo que requieren OpenSlide para su lectura.
# Se incluyen los formatos más comunes de escáneres de patología digital.
EXTENSIONES_WSI = {'.scn', '.svs', '.ndpi', '.mrxs', '.vms', '.bif', '.tif', '.tiff'}

# Extensiones que cv2.imread() soporta de forma nativa.
EXTENSIONES_ESTANDAR = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def es_formato_wsi(ruta: str) -> bool:
    """
    Determina si un archivo de imagen requiere OpenSlide para su lectura.

    Un archivo .tif/.tiff puede ser una imagen estándar o un WSI piramidal.
    Esta función intenta abrir con OpenSlide para decidir; si falla, asume
    que es un TIFF estándar legible por cv2.

    Args:
        ruta: Ruta absoluta al archivo de imagen.

    Returns:
        True si el archivo es un WSI que requiere OpenSlide, False en caso contrario.
    """
    ext = os.path.splitext(ruta)[1].lower()

    # Extensiones que SIEMPRE son WSI (Leica, Aperio, Hamamatsu, etc.)
    if ext in {'.scn', '.svs', '.ndpi', '.mrxs', '.vms', '.bif'}:
        return True

    # Para .tif/.tiff intentamos abrir con OpenSlide para detectar WSI piramidal
    if ext in {'.tif', '.tiff'}:
        try:
            import openslide
            slide = openslide.OpenSlide(ruta)
            es_piramidal = slide.level_count > 1
            slide.close()
            return es_piramidal
        except Exception:
            return False

    return False


def rgba_a_rgb(imagen_pil):
    """
    Convierte una imagen PIL de RGBA a RGB de forma eficiente.

    OpenSlide retorna imágenes en formato RGBA con canal alpha=255.
    La conversión a RGB ahorra ~25% de memoria al eliminar el canal alpha
    innecesario antes de convertir a numpy array.

    Args:
        imagen_pil: Imagen PIL en modo RGBA.

    Returns:
        Imagen PIL en modo RGB.
    """
    if imagen_pil.mode == 'RGBA':
        return imagen_pil.convert('RGB')
    return imagen_pil


def calcular_coordenadas_tiles(
    dimensiones: Tuple[int, int],
    tile_size: int = 512,
    overlap: int = 0
) -> Generator[Tuple[int, int], None, None]:
    """
    Genera coordenadas (x, y) para recorrer una imagen en tiles.

    El overlap permite que tiles adyacentes compartan píxeles en los bordes,
    lo cual es útil para análisis de segmentación donde las células en el
    borde de un tile podrían quedar cortadas.

    Args:
        dimensiones: Tupla (ancho, alto) de la imagen en píxeles.
        tile_size: Tamaño en píxeles de cada tile cuadrado.
        overlap: Número de píxeles de solapamiento entre tiles adyacentes.

    Yields:
        Tuplas (x, y) con la coordenada superior-izquierda de cada tile.
    """
    ancho, alto = dimensiones
    paso = tile_size - overlap

    if paso <= 0:
        raise ValueError(
            f"El overlap ({overlap}) debe ser menor que tile_size ({tile_size})."
        )

    for y in range(0, alto, paso):
        for x in range(0, ancho, paso):
            yield (x, y)


def filtrar_fondo(tile_np: np.ndarray, umbral_blancura: float = 0.85) -> bool:
    """
    Determina si un tile contiene tejido o es fondo blanco.

    En microscopía de fluorescencia y H&E, el fondo de la lámina es
    predominantemente blanco (alta intensidad en todos los canales RGB).
    Este filtro descarta tiles donde la mayoría de píxeles son blancos,
    evitando procesamiento innecesario de áreas vacías.

    Args:
        tile_np: Array numpy RGB del tile (shape: H x W x 3, dtype: uint8).
        umbral_blancura: Fracción de píxeles blancos (0.0-1.0) por encima de
                          la cual el tile se considera fondo. Default: 0.85.

    Returns:
        True si el tile contiene tejido (NO es fondo), False si es fondo.
    """
    if tile_np.size == 0:
        return False

    # Un píxel se considera "blanco" si TODOS sus canales RGB superan 220/255.
    # Esto cubre el fondo de láminas H&E y las áreas sin muestra en fluorescencia.
    es_blanco = np.all(tile_np > 220, axis=2)
    fraccion_blanca = np.mean(es_blanco)

    return fraccion_blanca < umbral_blancura


def calcular_factor_escala(
    nivel_origen: int,
    nivel_destino: int,
    downsamples: List[float]
) -> float:
    """
    Calcula el factor de escala entre dos niveles piramidales de un WSI.

    Útil para mapear coordenadas de un nivel de resolución baja (donde se
    detecta el tejido) al nivel de alta resolución (donde se extrae el ROI).

    Args:
        nivel_origen: Nivel piramidal de origen (ej: nivel de thumbnail).
        nivel_destino: Nivel piramidal de destino (ej: nivel 0 = máxima res.).
        downsamples: Lista de factores de submuestreo por nivel del slide.

    Returns:
        Factor multiplicativo para convertir coordenadas de origen a destino.
    """
    return downsamples[nivel_origen] / downsamples[nivel_destino]
