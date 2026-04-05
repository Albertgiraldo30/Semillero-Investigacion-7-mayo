"""
motor_apoptosis.py — Motor de análisis de apoptosis con soporte para Leica .SCN (WSI).

Integra OpenSlide para lectura eficiente de Whole Slide Images sin saturar la RAM,
manteniendo retrocompatibilidad total con imágenes estándar (JPG, PNG, BMP, TIFF plano)
leídas con OpenCV.

Clases:
    ProcesadorTirillas: Procesa imágenes de membranas del Proteome Profiler Array.
    AnalizadorApoptosis: Calcula Fold Change normalizado por Reference Spots.
    LectorWSI: Lector eficiente de WSI con extracción de tiles bajo demanda.

Dependencias:
    pip install openslide-python openslide-bin numpy opencv-python Pillow

Autor: Semillero Biomédica ITM — Detección de Cáncer
"""

import cv2
import logging
import os
import numpy as np
from typing import Optional, Dict, Tuple, Generator, List

from utils_wsi import (
    es_formato_wsi,
    rgba_a_rgb,
    calcular_coordenadas_tiles,
    filtrar_fondo,
    calcular_factor_escala,
)

# Dependencia opcional: OpenSlide solo se importa si hay archivos WSI
try:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator
    OPENSLIDE_DISPONIBLE = True
except ImportError:
    OPENSLIDE_DISPONIBLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(name)s]: %(message)s')
logger = logging.getLogger('MotorApoptosis')

# Umbral mínimo de intensidad para considerar que una proteína está presente.
EPSILON_INTENSIDAD = 1.0


# =============================================================================
# CLASE 1: Lector de Whole Slide Images (WSI)
# =============================================================================

class LectorWSI:
    """
    Lector eficiente de Whole Slide Images (WSI) usando OpenSlide.

    Diseñado para manejar archivos .scn de Leica (y otros formatos WSI) sin
    cargar la imagen completa en memoria. Utiliza la estructura piramidal del
    archivo para acceder a tiles bajo demanda.

    Uso con context manager:
        with LectorWSI('slide.scn') as lector:
            thumb = lector.obtener_thumbnail(1024)
            roi = lector.leer_roi(1000, 2000, 512, 512)

    Atributos:
        ruta (str): Ruta al archivo WSI.
        slide (openslide.OpenSlide): Handle al archivo WSI (lazy-loaded).
    """

    def __init__(self, ruta: str):
        """
        Inicializa el lector WSI.

        Args:
            ruta: Ruta absoluta al archivo WSI (.scn, .svs, .ndpi, etc.)

        Raises:
            ImportError: Si openslide-python no está instalado.
            FileNotFoundError: Si la ruta no existe.
        """
        if not OPENSLIDE_DISPONIBLE:
            raise ImportError(
                "openslide-python no está instalado. "
                "Instálelo con: pip install openslide-python openslide-bin"
            )

        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Archivo WSI no encontrado: {ruta}")

        self.ruta = ruta
        self._slide: Optional[openslide.OpenSlide] = None
        self._deepzoom: Optional[DeepZoomGenerator] = None

    @property
    def slide(self) -> 'openslide.OpenSlide':
        """Acceso lazy al handle de OpenSlide. Se abre solo cuando se necesita."""
        if self._slide is None:
            logger.info(f"Abriendo WSI: {os.path.basename(self.ruta)}")
            self._slide = openslide.OpenSlide(self.ruta)
        return self._slide

    @property
    def deepzoom(self) -> 'DeepZoomGenerator':
        """Generador DeepZoom para acceso eficiente por tiles indexados."""
        if self._deepzoom is None:
            self._deepzoom = DeepZoomGenerator(
                self.slide,
                tile_size=254,   # 254 + 2 de overlap = 256 px por tile
                overlap=1,
                limit_bounds=True
            )
        return self._deepzoom

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cerrar()
        return False

    def cerrar(self):
        """Libera recursos del handle de OpenSlide."""
        if self._slide is not None:
            self._slide.close()
            self._slide = None
            self._deepzoom = None
            logger.info("WSI cerrado correctamente.")

    # -------------------------------------------------------------------------
    # Metadatos
    # -------------------------------------------------------------------------

    def obtener_metadata(self) -> Dict:
        """
        Extrae metadatos del WSI: dimensiones, niveles, resolución óptica.

        Returns:
            Diccionario con metadata del slide:
                - dimensiones_nivel_0: (ancho, alto) en máxima resolución
                - niveles: número de niveles piramidales
                - dimensiones_por_nivel: [(ancho, alto), ...] por cada nivel
                - downsamples: [factor, ...] por cada nivel
                - mpp_x, mpp_y: micrones por píxel (si disponible)
                - propiedades_leica: metadatos específicos de Leica (si aplica)
        """
        meta = {
            'dimensiones_nivel_0': self.slide.dimensions,
            'niveles': self.slide.level_count,
            'dimensiones_por_nivel': list(self.slide.level_dimensions),
            'downsamples': list(self.slide.level_downsamples),
        }

        # Resolución óptica (micrones por píxel)
        props = self.slide.properties
        meta['mpp_x'] = props.get(openslide.PROPERTY_NAME_MPP_X)
        meta['mpp_y'] = props.get(openslide.PROPERTY_NAME_MPP_Y)

        # Propiedades específicas de Leica .scn
        leica_props = {k: v for k, v in props.items() if 'leica' in k.lower()}
        if leica_props:
            meta['propiedades_leica'] = leica_props

        return meta

    # -------------------------------------------------------------------------
    # Thumbnails y vistas previas
    # -------------------------------------------------------------------------

    def obtener_thumbnail(self, max_dim: int = 1024) -> np.ndarray:
        """
        Genera una vista previa de baja resolución del slide completo.

        Utiliza el método nativo de OpenSlide que selecciona automáticamente
        el nivel piramidal más eficiente para generar el thumbnail.

        Args:
            max_dim: Dimensión máxima (ancho o alto) del thumbnail en píxeles.

        Returns:
            Array numpy RGB del thumbnail (HxWx3, uint8).
        """
        thumb_pil = self.slide.get_thumbnail((max_dim, max_dim))
        thumb_rgb = rgba_a_rgb(thumb_pil)
        return np.array(thumb_rgb, dtype=np.uint8)

    # -------------------------------------------------------------------------
    # Lectura de ROIs
    # -------------------------------------------------------------------------

    def leer_roi(
        self,
        x: int,
        y: int,
        ancho: int,
        alto: int,
        nivel: int = 0
    ) -> np.ndarray:
        """
        Extrae una región de interés (ROI) del WSI.

        IMPORTANTE: Las coordenadas (x, y) SIEMPRE se expresan en el sistema
        de coordenadas del nivel 0 (máxima resolución), independientemente del
        nivel que se solicite. El tamaño (ancho, alto) sí corresponde al nivel
        solicitado.

        Args:
            x: Coordenada X superior-izquierda (en sistema del nivel 0).
            y: Coordenada Y superior-izquierda (en sistema del nivel 0).
            ancho: Ancho del ROI en píxeles (en el nivel solicitado).
            alto: Alto del ROI en píxeles (en el nivel solicitado).
            nivel: Nivel piramidal a leer (0 = máxima resolución).

        Returns:
            Array numpy RGB del ROI (HxWx3, uint8).

        Raises:
            ValueError: Si el nivel solicitado no existe.
        """
        if nivel >= self.slide.level_count:
            raise ValueError(
                f"Nivel {nivel} no existe. Este WSI tiene {self.slide.level_count} niveles."
            )

        region_pil = self.slide.read_region(
            location=(x, y),
            level=nivel,
            size=(ancho, alto)
        )

        region_rgb = rgba_a_rgb(region_pil)
        return np.array(region_rgb, dtype=np.uint8)

    # -------------------------------------------------------------------------
    # Máscara de tejido
    # -------------------------------------------------------------------------

    def generar_mascara_tejido(
        self,
        umbral_blanco: int = 220,
        tamano_thumb: int = 1024
    ) -> Tuple[np.ndarray, float]:
        """
        Genera una máscara binaria de tejido vs. fondo a partir de un thumbnail.

        Algoritmo:
            1. Obtener thumbnail de baja resolución (rápido, ~1 MB en RAM).
            2. Convertir a escala de grises.
            3. Aplicar umbralización inversa (tejido=blanco, fondo=negro).
            4. Operaciones morfológicas para limpiar ruido.

        Args:
            umbral_blanco: Valor de gris (0-255) por encima del cual se
                           considera fondo blanco.
            tamano_thumb: Tamaño del thumbnail para calcular la máscara.

        Returns:
            Tupla (mascara, escala):
                - mascara: Array binario uint8 (255=tejido, 0=fondo).
                - escala: Factor para convertir coordenadas de la máscara
                           al sistema de coordenadas del nivel 0.
        """
        thumb = self.obtener_thumbnail(tamano_thumb)
        gris = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)

        # Umbralización: el tejido tiene intensidad baja, el fondo alta
        _, mascara = cv2.threshold(gris, umbral_blanco, 255, cv2.THRESH_BINARY_INV)

        # Limpiar ruido con operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

        # Factor de escala: thumbnail → nivel 0
        dim_0 = self.slide.dimensions
        escala = max(dim_0[0] / thumb.shape[1], dim_0[1] / thumb.shape[0])

        logger.info(
            f"Máscara de tejido generada. "
            f"Cobertura de tejido: {np.mean(mascara > 0) * 100:.1f}%"
        )

        return mascara, escala

    # -------------------------------------------------------------------------
    # Iterador de tiles (generador — memoria constante)
    # -------------------------------------------------------------------------

    def iterar_tiles(
        self,
        tile_size: int = 512,
        nivel: int = 0,
        solo_tejido: bool = True,
        umbral_blancura: float = 0.85,
        overlap: int = 0
    ) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """
        Generador que produce tiles del WSI bajo demanda (lazy iteration).

        GARANTÍA DE MEMORIA: Este método usa yield, por lo que NUNCA hay más
        de un tile cargado en RAM simultáneamente. El consumo de memoria es
        constante sin importar el tamaño del WSI: ~(tile_size² × 3) bytes.

        Para un tile_size=512: ~768 KB por tile en RAM.
        Para un tile_size=1024: ~3 MB por tile en RAM.

        Args:
            tile_size: Tamaño en px del tile cuadrado (512 o 1024 recomendado).
            nivel: Nivel piramidal a leer.
            solo_tejido: Si True, filtra tiles que son fondo blanco.
            umbral_blancura: Fracción de píxeles blancos para descartar tile.
            overlap: Píxeles de solapamiento entre tiles adyacentes.

        Yields:
            Tupla (x, y, tile_np):
                - x, y: Coordenadas del tile en sistema del nivel 0.
                - tile_np: Array numpy RGB del tile (HxWx3, uint8).
        """
        level_dims = self.slide.level_dimensions[nivel]
        downsample = self.slide.level_downsamples[nivel]
        total_tiles = 0
        tiles_tejido = 0

        for x_nivel, y_nivel in calcular_coordenadas_tiles(
            dimensiones=level_dims,
            tile_size=tile_size,
            overlap=overlap
        ):
            total_tiles += 1

            # Coordenadas en sistema del nivel 0 (requerido por read_region)
            x_0 = int(x_nivel * downsample)
            y_0 = int(y_nivel * downsample)

            # Leer tile desde el WSI
            tile_pil = self.slide.read_region(
                location=(x_0, y_0),
                level=nivel,
                size=(tile_size, tile_size)
            )
            tile_rgb = rgba_a_rgb(tile_pil)
            tile_np = np.array(tile_rgb, dtype=np.uint8)

            # Filtrar fondo si se solicita
            if solo_tejido and not filtrar_fondo(tile_np, umbral_blancura):
                continue

            tiles_tejido += 1
            yield x_0, y_0, tile_np

        logger.info(
            f"Iteración completada: {tiles_tejido}/{total_tiles} tiles con tejido "
            f"(nivel {nivel}, tile_size={tile_size}px)"
        )

    def extraer_tiles_centro(
        self,
        num_tiles: int = 9,
        tile_size: int = 512,
        nivel: int = 0
    ) -> List[Tuple[int, int, np.ndarray]]:
        """
        Extrae tiles del centro geométrico del tejido detectado.

        Útil cuando no hay anotaciones del patólogo y se necesita una muestra
        representativa del tejido para análisis rápido.

        Args:
            num_tiles: Número de tiles a extraer (formará un grid NxN centrado).
            tile_size: Tamaño de cada tile en píxeles.
            nivel: Nivel piramidal a leer.

        Returns:
            Lista de (x, y, tile_np) con los tiles del centro del tejido.
        """
        mascara, escala = self.generar_mascara_tejido()

        # Encontrar el centro de masa del tejido
        momentos = cv2.moments(mascara)
        if momentos['m00'] == 0:
            logger.warning("No se detectó tejido. Usando centro de la imagen.")
            centro_x = self.slide.dimensions[0] // 2
            centro_y = self.slide.dimensions[1] // 2
        else:
            # Centro en coordenadas del thumbnail → escalar a nivel 0
            cx_thumb = momentos['m10'] / momentos['m00']
            cy_thumb = momentos['m01'] / momentos['m00']
            centro_x = int(cx_thumb * escala)
            centro_y = int(cy_thumb * escala)

        logger.info(f"Centro del tejido detectado en ({centro_x}, {centro_y}) nivel 0")

        # Calcular grid de tiles centrado en el tejido
        lado_grid = int(np.ceil(np.sqrt(num_tiles)))
        offset_total = lado_grid * tile_size
        inicio_x = centro_x - offset_total // 2
        inicio_y = centro_y - offset_total // 2

        # Asegurar que no nos salgamos de los límites
        dim_nivel = self.slide.level_dimensions[nivel]
        downsample = self.slide.level_downsamples[nivel]
        max_x = int(dim_nivel[0] * downsample) - tile_size
        max_y = int(dim_nivel[1] * downsample) - tile_size
        inicio_x = max(0, min(inicio_x, max_x))
        inicio_y = max(0, min(inicio_y, max_y))

        tiles = []
        for i in range(lado_grid):
            for j in range(lado_grid):
                if len(tiles) >= num_tiles:
                    break
                x = inicio_x + j * tile_size
                y = inicio_y + i * tile_size
                tile_np = self.leer_roi(x, y, tile_size, tile_size, nivel)

                # Solo incluir tiles que contengan tejido
                if filtrar_fondo(tile_np):
                    tiles.append((x, y, tile_np))

        logger.info(f"Extraídos {len(tiles)} tiles del centro del tejido.")
        return tiles


# =============================================================================
# CLASE 2: Procesador de Tirillas (Proteome Profiler Array)
# =============================================================================

class ProcesadorTirillas:
    """
    Procesador de imágenes de membranas del Proteome Profiler Human Apoptosis Array.

    Soporta lectura de imágenes en formatos estándar (JPG, PNG, TIFF plano) via
    OpenCV, y formatos WSI (SCN, SVS, NDPI) via OpenSlide.

    Para imágenes WSI, se extrae automáticamente un thumbnail de alta calidad
    que es procesado como una imagen estándar. Las membranas del Proteome
    Profiler no requieren resolución de nivel 0 del WSI.
    """

    def __init__(
        self,
        ruta_imagen: str,
        umbral_canny_min: int = 50,
        umbral_canny_max: int = 150,
        epsilon_poly: float = 0.02
    ):
        """
        Inicializa el ProcesadorTirillas con la ruta y parámetros de procesamiento.

        Args:
            ruta_imagen: Ruta al archivo de la imagen de la membrana.
            umbral_canny_min: Umbral mínimo para el detector de bordes Canny.
            umbral_canny_max: Umbral máximo para el detector de bordes Canny.
            epsilon_poly: Tolerancia para la aproximación poligonal (fracción del perímetro).
        """
        self.ruta_imagen = ruta_imagen
        self.umbral_canny_min = umbral_canny_min
        self.umbral_canny_max = umbral_canny_max
        self.epsilon_poly = epsilon_poly
        self._lector_wsi: Optional[LectorWSI] = None

        # Enrutamiento inteligente basado en extensión del archivo
        # Solo extensiones exclusivamente WSI van a OpenSlide.
        # Formatos estándar (.jpg, .png, .tif, .bmp) SIEMPRE usan cv2.
        ext = os.path.splitext(ruta_imagen)[1].lower()
        self._EXTENSIONES_WSI = {'.scn', '.svs', '.ndpi', '.mrxs', '.vms', '.bif'}
        self._EXTENSIONES_OPENCV = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self._es_wsi = ext in self._EXTENSIONES_WSI

    def alinear_tirilla(self, imagen: np.ndarray) -> np.ndarray:
        """
        Detecta el contorno rectangular de la membrana y aplica transformación
        de perspectiva para enderezarla y recortarla.

        A diferencia de cv2.boundingRect() (que produce un recorte ortogonal con
        esquinas vacías si la membrana está rotada), getPerspectiveTransform()
        mapea las 4 esquinas reales al rectángulo destino, eliminando la rotación.

        Args:
            imagen: La imagen original BGR (numpy array).

        Returns:
            Imagen recortada y enderezada de la membrana, a color.
        """
        img_gris = cv2.cvtColor(imagen.copy(), cv2.COLOR_BGR2GRAY)
        img_difuminada = cv2.GaussianBlur(img_gris, (5, 5), 0)
        bordes = cv2.Canny(img_difuminada, self.umbral_canny_min, self.umbral_canny_max)

        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            logger.warning("No se detectaron contornos en la imagen. Se retorna la imagen original.")
            return imagen

        contornos_grandes = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
        contorno_tirilla = None

        for contorno in contornos_grandes:
            perimetro = cv2.arcLength(contorno, True)
            aproximacion = cv2.approxPolyDP(contorno, self.epsilon_poly * perimetro, True)
            if len(aproximacion) == 4:
                contorno_tirilla = aproximacion
                break

        if contorno_tirilla is None:
            logger.warning("No se encontró contorno de 4 lados. Se retorna la imagen original.")
            return imagen

        # Transformación de perspectiva
        puntos = contorno_tirilla.reshape(4, 2).astype(np.float32)
        puntos_ordenados = self._ordenar_puntos(puntos)

        ancho = int(max(
            np.linalg.norm(puntos_ordenados[1] - puntos_ordenados[0]),
            np.linalg.norm(puntos_ordenados[2] - puntos_ordenados[3])
        ))
        alto = int(max(
            np.linalg.norm(puntos_ordenados[3] - puntos_ordenados[0]),
            np.linalg.norm(puntos_ordenados[2] - puntos_ordenados[1])
        ))

        destino = np.array([
            [0, 0], [ancho - 1, 0],
            [ancho - 1, alto - 1], [0, alto - 1]
        ], dtype=np.float32)

        matriz = cv2.getPerspectiveTransform(puntos_ordenados, destino)
        return cv2.warpPerspective(imagen, matriz, (ancho, alto))

    @staticmethod
    def _ordenar_puntos(puntos: np.ndarray) -> np.ndarray:
        """
        Ordena 4 puntos: [superior-izq, superior-der, inferior-der, inferior-izq].

        La suma (x+y) identifica las esquinas diagonalmente opuestas;
        la diferencia (x-y) distingue las otras dos.
        """
        suma = puntos.sum(axis=1)
        diferencia = np.diff(puntos, axis=1).flatten()
        return np.array([
            puntos[np.argmin(suma)],
            puntos[np.argmin(diferencia)],
            puntos[np.argmax(suma)],
            puntos[np.argmax(diferencia)],
        ], dtype=np.float32)

    def _cargar_imagen(self) -> np.ndarray:
        """
        Carga la imagen con enrutamiento inteligente basado en extensión:

        - .scn, .svs, .ndpi  → OpenSlide (LectorWSI → thumbnail 4096px)
        - .jpg, .png, .tif   → OpenCV directo (cv2.imread)

        Este enrutamiento garantiza que las imágenes estándar del laboratorio
        siempre funcionen con cv2, incluso si OpenSlide no está instalado
        o los archivos .scn están corruptos.

        Returns:
            Imagen BGR como numpy array.

        Raises:
            FileNotFoundError: Si la imagen no se puede cargar.
            ValueError: Si la extensión del archivo no es soportada.
        """
        ext = os.path.splitext(self.ruta_imagen)[1].lower()

        if self._es_wsi:
            # ── Ruta WSI: OpenSlide ──
            logger.info(f"Detectado formato WSI ({ext}). Usando OpenSlide...")

            if not OPENSLIDE_DISPONIBLE:
                raise ImportError(
                    f"El archivo '{os.path.basename(self.ruta_imagen)}' es formato WSI ({ext}) "
                    f"pero openslide-python no está instalado.\n"
                    f"Instálelo con: pip install openslide-python openslide-bin"
                )

            self._lector_wsi = LectorWSI(self.ruta_imagen)
            thumbnail_rgb = self._lector_wsi.obtener_thumbnail(max_dim=4096)
            self._lector_wsi.cerrar()
            # Convertir RGB (OpenSlide) → BGR (OpenCV)
            imagen_bgr = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR)
            return imagen_bgr

        elif ext in self._EXTENSIONES_OPENCV:
            # ── Ruta estándar: OpenCV directo ──
            logger.info(f"Formato estándar ({ext}). Usando OpenCV directo...")
            imagen = cv2.imread(self.ruta_imagen)
            if imagen is None:
                raise FileNotFoundError(
                    f"No se pudo cargar la imagen en: {self.ruta_imagen}"
                )
            return imagen

        else:
            raise ValueError(
                f"Extensión '{ext}' no soportada. "
                f"Formatos válidos: {self._EXTENSIONES_OPENCV | self._EXTENSIONES_WSI}"
            )

    def preprocesar_imagen(self) -> np.ndarray:
        """
        Lee la imagen (WSI o estándar), alinea la membrana, convierte a escala
        de grises e invierte.

        La inversión es necesaria porque en la membrana original los puntos de
        proteína aparecen oscuros sobre fondo claro. Al invertir, el fondo queda
        en 0 (negro) y los puntos en valores altos (blancos), de modo que la
        intensidad promedio de un ROI es directamente proporcional a la
        concentración de proteína.

        Returns:
            Imagen en escala de grises invertida (numpy array uint8).
        """
        imagen = self._cargar_imagen()
        imagen_alineada = self.alinear_tirilla(imagen)
        imagen_gris = cv2.cvtColor(imagen_alineada, cv2.COLOR_BGR2GRAY)
        imagen_invertida = cv2.bitwise_not(imagen_gris)
        return imagen_invertida

    def extraer_intensidad_puntos(
        self,
        imagen_procesada: np.ndarray,
        rois: Dict[str, Tuple[int, int, int, int]]
    ) -> Dict[str, Optional[float]]:
        """
        Extrae la intensidad promedio (0–255) de cada región de interés (ROI).

        Args:
            imagen_procesada: Imagen en escala de grises invertida.
            rois: Diccionario {nombre: (x, y, ancho, alto)}.

        Returns:
            Diccionario {nombre: intensidad_promedio} donde None indica ROI inválido.
            Los ROIs inválidos se reportan como None (no como 0.0) para que el
            analizador distinga "ausencia real" de "dato no medible".
        """
        alto_imagen, ancho_imagen = imagen_procesada.shape[:2]
        resultados = {}

        for nombre, coordenadas in rois.items():
            x, y, ancho, alto = coordenadas

            # Validar límites antes de recortar
            if x < 0 or y < 0 or (x + ancho) > ancho_imagen or (y + alto) > alto_imagen:
                logger.warning(
                    f"ROI '{nombre}' ({x},{y},{ancho},{alto}) excede los límites de la imagen "
                    f"({ancho_imagen}x{alto_imagen}). Se registra como None."
                )
                resultados[nombre] = None
                continue

            recorte = imagen_procesada[y:y + alto, x:x + ancho]

            if recorte.size == 0:
                logger.warning(f"ROI '{nombre}' produce un recorte vacío. Se registra como None.")
                resultados[nombre] = None
                continue

            resultados[nombre] = round(cv2.mean(recorte)[0], 2)

        return resultados


# =============================================================================
# CLASE 3: Analizador de Apoptosis (Fold Change)
# =============================================================================

class AnalizadorApoptosis:
    """
    Analizador de resultados del Proteome Profiler Human Apoptosis Array
    usando Densidad Óptica Relativa normalizada (Fold Change).

    El Fold Change se calcula como Tratamiento / Control para cada proteína,
    normalizado por los Reference Spots del array para eliminar variaciones
    técnicas entre membranas.
    """

    def __init__(
        self,
        pro_apoptoticas: Optional[List[str]] = None,
        anti_apoptoticas: Optional[List[str]] = None
    ):
        """
        Inicializa el analizador con las listas de proteínas por categoría biológica.

        Args:
            pro_apoptoticas: Nombres de proteínas pro-apoptóticas.
            anti_apoptoticas: Nombres de proteínas anti-apoptóticas.
        """
        self.pro_apoptoticas = pro_apoptoticas or [
            'Bax', 'Bad', 'Cytochrome c', 'Caspase-3',
            'Cleaved Caspase-3', 'SMAC/Diablo', 'FADD',
            'Phospho-p53 (S15)', 'Phospho-p53 (S46)', 'Phospho-p53 (S392)',
            'Phospho-Rad17 (S635)', 'DR4', 'DR5',
            'TRAIL R1/DR4', 'TRAIL R2/DR5',
            'TNF RI/TNFRSF1A', 'Fas/TNFRSF6/CD95',
            'Pro-Caspase-3', 'Caspase-8/10',
        ]
        self.anti_apoptoticas = anti_apoptoticas or [
            'Bcl-2', 'Bcl-xL', 'Survivin', 'XIAP',
            'cIAP-1', 'cIAP-2', 'Livin',
            'Claspin', 'Clusterin',
            'HIF-1α', 'HO-1/HMOX1/HSP32', 'HO-2/HMOX2',
            'HSP27', 'HSP60', 'HSP70',
            'p21/CIP1/CDKN1A', 'p27/Kip1',
        ]

    def normalizar_por_referencia(
        self,
        intensidades: Dict[str, Optional[float]],
        nombres_referencia: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        Normaliza las intensidades dividiéndolas por el promedio de los
        Reference Spots positivos del array.

        Los arrays Proteome Profiler incluyen duplicados de puntos de referencia
        positivos (manchas de anticuerpo conocido) en cada membrana. Normalizar
        por su promedio elimina las variaciones técnicas de exposición entre
        membranas (distintos tiempos de revelado, cantidad de muestra cargada).

        Args:
            intensidades: Intensidades crudas {nombre: valor}.
            nombres_referencia: Lista de nombres de los Reference Spots.

        Returns:
            Intensidades normalizadas. Si no hay referencias válidas, retorna
            las intensidades originales con una advertencia.
        """
        valores_ref = [
            intensidades[n] for n in nombres_referencia
            if n in intensidades and intensidades[n] is not None and intensidades[n] > EPSILON_INTENSIDAD
        ]

        if not valores_ref:
            logger.warning(
                "No se encontraron puntos de referencia válidos para normalizar. "
                "Se usarán las intensidades crudas."
            )
            return intensidades

        promedio_referencia = np.mean(valores_ref)
        logger.info(f"Normalizando por promedio de referencia: {promedio_referencia:.2f}")

        normalizadas = {}
        for nombre, valor in intensidades.items():
            if valor is None:
                normalizadas[nombre] = None
            else:
                normalizadas[nombre] = round(valor / promedio_referencia, 4)

        return normalizadas

    def calcular_fold_change(
        self,
        intensidades_control: Dict[str, Optional[float]],
        intensidades_tratamiento: Dict[str, Optional[float]]
    ) -> Dict:
        """
        Calcula el Fold Change (Tratamiento / Control) para cada proteína.

        Reglas de cálculo:
        - Control > ε y Tratamiento > ε  → Fold = Tratamiento / Control
        - Control < ε y Tratamiento > ε  → Expresión de novo: float('inf')
        - Control > ε y Tratamiento < ε  → Silenciada: 0.0
        - Ambas < ε                       → Ausente: EXCLUIDA del reporte
        - Cualquier valor None            → Inválida: EXCLUIDA del reporte

        Args:
            intensidades_control: Intensidades normalizadas del control.
            intensidades_tratamiento: Intensidades normalizadas del tratamiento.

        Returns:
            Reporte estructurado {categoría: {proteína: fold_change}},
            más un campo 'Resumen' con métricas derivadas.
        """
        reporte = {
            'Pro-apoptóticas': {},
            'Anti-apoptóticas': {},
            'No Clasificadas (Otras)': {},
        }

        todas = set(intensidades_control.keys()) | set(intensidades_tratamiento.keys())

        for proteina in todas:
            val_ctrl = intensidades_control.get(proteina)
            val_trat = intensidades_tratamiento.get(proteina)

            # Excluir mediciones inválidas (None)
            if val_ctrl is None or val_trat is None:
                logger.warning(f"Proteína '{proteina}' tiene medición inválida. Excluida.")
                continue

            ctrl_ausente = val_ctrl < EPSILON_INTENSIDAD
            trat_ausente = val_trat < EPSILON_INTENSIDAD

            # Excluir proteínas ausentes en ambas condiciones
            if ctrl_ausente and trat_ausente:
                logger.info(f"Proteína '{proteina}' ausente en ambas condiciones. Excluida.")
                continue

            if ctrl_ausente:
                fold_change = float('inf')
                logger.info(f"'{proteina}': expresión de novo (ctrl=0, trat={val_trat:.4f}).")
            elif trat_ausente:
                fold_change = 0.0
                logger.info(f"'{proteina}': silenciada completamente (trat≈0).")
            else:
                fold_change = round(val_trat / val_ctrl, 4)

            # Clasificar por categoría biológica
            if proteina in self.pro_apoptoticas:
                reporte['Pro-apoptóticas'][proteina] = fold_change
            elif proteina in self.anti_apoptoticas:
                reporte['Anti-apoptóticas'][proteina] = fold_change
            else:
                reporte['No Clasificadas (Otras)'][proteina] = fold_change

        reporte['Resumen'] = self._calcular_resumen(reporte)
        return reporte

    def _calcular_resumen(self, reporte: Dict) -> Dict:
        """
        Calcula métricas derivadas del reporte de fold change.

        Incluye:
        - Top 5 proteínas upreguladas y downreguladas
        - Ratio pro/anti-apoptótico (>1 indica tendencia pro-apoptótica)
        - Interpretación clínica automatizada

        Args:
            reporte: Reporte con categorías y sus fold changes.

        Returns:
            Diccionario con métricas de resumen.
        """
        def fold_finito(valor):
            return valor if valor != float('inf') else None

        todos_folds = {
            **reporte.get('Pro-apoptóticas', {}),
            **reporte.get('Anti-apoptóticas', {}),
            **reporte.get('No Clasificadas (Otras)', {}),
        }

        folds_finitos = {k: v for k, v in todos_folds.items() if fold_finito(v) is not None}

        upreguladas = sorted(
            [(p, f) for p, f in folds_finitos.items() if f > 1.0],
            key=lambda x: x[1], reverse=True
        )
        downreguladas = sorted(
            [(p, f) for p, f in folds_finitos.items() if f < 1.0],
            key=lambda x: x[1]
        )

        # Ratio pro/anti-apoptótico
        pro_folds = [f for f in reporte.get('Pro-apoptóticas', {}).values()
                     if fold_finito(f) is not None]
        anti_folds = [f for f in reporte.get('Anti-apoptóticas', {}).values()
                      if fold_finito(f) is not None]

        media_pro = round(np.mean(pro_folds), 4) if pro_folds else None
        media_anti = round(np.mean(anti_folds), 4) if anti_folds else None

        if media_pro is not None and media_anti is not None and media_anti > EPSILON_INTENSIDAD:
            ratio_pro_anti = round(media_pro / media_anti, 4)
        else:
            ratio_pro_anti = None

        return {
            'total_proteinas_analizadas': len(todos_folds),
            'upreguladas': [{'proteina': p, 'fold': f} for p, f in upreguladas[:5]],
            'downreguladas': [{'proteina': p, 'fold': f} for p, f in downreguladas[:5]],
            'ratio_pro_anti_apoptotico': ratio_pro_anti,
            'interpretacion': (
                'Tendencia pro-apoptótica (el tratamiento favorece la muerte celular)'
                if ratio_pro_anti and ratio_pro_anti > 1.0
                else 'Tendencia anti-apoptótica (el tratamiento favorece la supervivencia)'
                if ratio_pro_anti and ratio_pro_anti < 1.0
                else 'No determinada (datos insuficientes)'
            )
        }


# =============================================================================
# Ejemplo de uso
# =============================================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Motor de Apoptosis — Semillero ITM")
    print("=" * 60)

    # --- Ejemplo 1: Lectura de WSI (.scn) ---
    # Descomenta las siguientes líneas cuando tengas un archivo .scn real:

    # with LectorWSI('muestra.scn') as lector:
    #     meta = lector.obtener_metadata()
    #     print(f"Dimensiones: {meta['dimensiones_nivel_0']}")
    #     print(f"Niveles piramidales: {meta['niveles']}")
    #     print(f"Micrones/px: {meta.get('mpp_x', 'N/A')}")
    #
    #     # Extraer tiles del centro del tejido
    #     tiles_centro = lector.extraer_tiles_centro(num_tiles=9, tile_size=512)
    #     for i, (x, y, tile) in enumerate(tiles_centro):
    #         cv2.imwrite(f"tile_centro_{i}.png", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    #         print(f"  Tile {i}: ({x}, {y}) — shape: {tile.shape}")
    #
    #     # Iterar TODOS los tiles (generador — memoria constante)
    #     for x, y, tile in lector.iterar_tiles(tile_size=1024, solo_tejido=True):
    #         # Procesar tile aquí (ej: detección de apoptosis con ML)
    #         pass

    # --- Ejemplo 2: Análisis de membranas del Proteome Profiler ---

    # procesador_ctrl = ProcesadorTirillas("membrana_control.jpg")  # o .scn
    # procesador_trat = ProcesadorTirillas("membrana_tratada.jpg")
    #
    # img_ctrl = procesador_ctrl.preprocesar_imagen()
    # img_trat = procesador_trat.preprocesar_imagen()
    #
    # cuadricula = {
    #     'Reference_1': (10,  10, 15, 15),
    #     'Reference_2': (30,  10, 15, 15),
    #     'Bax':         (100, 200, 20, 20),
    #     'Caspase-3':   (150, 200, 20, 20),
    #     'Bcl-2':       (10,  10, 20, 20),
    #     'Bad':         (50,  50, 20, 20),
    # }
    # REFS = ['Reference_1', 'Reference_2']
    #
    # datos_ctrl = procesador_ctrl.extraer_intensidad_puntos(img_ctrl, cuadricula)
    # datos_trat = procesador_trat.extraer_intensidad_puntos(img_trat, cuadricula)
    #
    # analizador = AnalizadorApoptosis(
    #     pro_apoptoticas=['Bax', 'Caspase-3', 'Bad'],
    #     anti_apoptoticas=['Bcl-2']
    # )
    #
    # ctrl_norm = analizador.normalizar_por_referencia(datos_ctrl, REFS)
    # trat_norm = analizador.normalizar_por_referencia(datos_trat, REFS)
    #
    # reporte = analizador.calcular_fold_change(ctrl_norm, trat_norm)
    # print(json.dumps(reporte, indent=4, ensure_ascii=False))

    print("\nMotor cargado correctamente. Listo para análisis.")
