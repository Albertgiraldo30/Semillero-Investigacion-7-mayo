"""
motor_apoptosis.py — Motor de análisis para Proteome Profiler ARY009
Detecta spots en la imagen completa, los agrupa en strips, y mapea a proteínas.
"""
import cv2
import logging
import os
import numpy as np
from typing import Optional, Dict, Tuple, List

from mapa_array import (
    MAPA_PROTEINAS, PROTEINAS_UNICAS, PRO_APOPTOTICAS, ANTI_APOPTOTICAS,
    COORDS_REFERENCIA, COORDS_PBS, PARES_DUPLICADOS, T_COLS,
)
import gridfit_ary009

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(name)s]: %(message)s')
logger = logging.getLogger('MotorApoptosis')

EPSILON = 1.0


# =====================================================================
# 1. Detección robusta de spots (Black-Hat)
# =====================================================================


def detectar_spots_en_strip(strip_bgr: np.ndarray) -> List[Dict]:
    """
    Usa la transformación morfológica Black-Hat para aislar 
    puntos pequeños y oscuros sobre un fondo claro, ignorando sombras globales.
    """
    gris = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Denoising ligero
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    # 2. Black-Hat transform: extrae elementos oscuros (más pequeños que el kernel)
    # Un spot típico en microarray mide ~5-15 píxeles de diámetro
    tam_kernel = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_kernel, tam_kernel))
    blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)

    # Ahora 'blackhat' tiene los spots como píxeles brillantes sobre fondo negro
    # 3. Umbralización sobre el resultado
    _, binaria = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)

    # 4. Limpieza (quitar ruido de 1 o 2 píxeles)
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_clean, iterations=1)

    # Encontrar contornos
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Invertir una sola vez fuera del loop (255=máx intensidad de proteína)
    invertida = cv2.bitwise_not(gris)

    spots = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        # Filtrar manchas demasiado pequeñas o enormes
        if area < 5 or area > 800:
            continue
            
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Calcular centro
        momentos = cv2.moments(contorno)
        if momentos['m00'] > 0:
            cx = int(momentos['m10'] / momentos['m00'])
            cy = int(momentos['m01'] / momentos['m00'])
        else:
            cx, cy = x + w // 2, y + h // 2
        mascara = np.zeros(gris.shape, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        intensidad = cv2.mean(invertida, mask=mascara)[0]
        
        spots.append({
            'bbox': (x, y, w, h),
            'centro': (cx, cy),
            'area': round(area, 1),
            'intensidad': round(intensidad, 2),
        })
        
    logger.info(f"Detectados {len(spots)} spots usando Black-Hat.")
    return spots


# =====================================================================
# 3. Mapeo de spots al grid del array (columnas A-E, filas 1-24)
# =====================================================================

def mapear_spots_a_grid(spots: List[Dict]) -> List[Dict]:
    """
    Asigna coordenadas (columna, fila) y nombre de proteína a cada spot.
    Utiliza una cuadrícula geométrica asumiendo que los puntos superior/inferior
    y los laterales definen los bordes del grid biológico.

    Layout del array ARY009 (vista frontal):
    - Columna E: izquierda
    - Columna A: derecha
    - Filas 1 a 24
    """
    if not spots:
        return []

    # Encontrar los extremos geométricos de este strip
    xs = [s['centro'][0] for s in spots]
    ys = [s['centro'][1] for s in spots]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # El array tiene 24 filas (1 a 24) -> 23 espacios entre ellas
    # y 5 columnas (E, D, C, B, A) -> 4 espacios entre ellas
    alto_grid = max(1, y_max - y_min)
    ancho_grid = max(1, x_max - x_min)
    
    paso_y = alto_grid / 23.0
    paso_x = ancho_grid / 4.0
    
    # De izquierda a derecha las columnas son E, D, C, B, A
    columnas_letras = ['E', 'D', 'C', 'B', 'A']
    
    mapeados = []
    for spot in spots:
        cx, cy = spot['centro']
        
        # Calcular fila (0 a 23 -> 1 a 24)
        idx_fila = int(round((cy - y_min) / paso_y))
        idx_fila = max(0, min(23, idx_fila))
        fila_num = idx_fila + 1
        
        # Calcular columna (0 a 4 -> E a A)
        idx_col = int(round((cx - x_min) / paso_x))
        idx_col = max(0, min(4, idx_col))
        col_letra = columnas_letras[idx_col]
        
        coord_tuple = (col_letra, fila_num)
        coord_str = f"{col_letra}{fila_num}"
        nombre = MAPA_PROTEINAS.get(coord_tuple, f"? ({coord_str})")
        
        spot['columna'] = col_letra
        spot['fila'] = fila_num
        spot['coordenada'] = coord_str
        spot['proteina'] = nombre
        mapeados.append(spot)
        
    logger.info(f"Mapeados {len(mapeados)} spots usando grid geométrico.")
    return mapeados


# =====================================================================
# 4. Cuantificación: promediar duplicados, restar PBS, normalizar
# =====================================================================

def cuantificar_proteinas(spots_mapeados: List[Dict]) -> Dict:
    """
    1. Promedia duplicados de cada proteína
    2. Resta el fondo PBS (D23/D24)
    3. Normaliza por Reference Spots
    """
    # Agrupar por proteína
    grupos = {}
    for spot in spots_mapeados:
        nombre = spot.get('proteina', '?')
        if nombre.startswith("?"):
            continue
        grupos.setdefault(nombre, []).append(spot['intensidad'])

    promedios = {n: round(np.mean(v), 2) for n, v in grupos.items()}

    # PBS
    pbs_val = promedios.get("PBS", 0.0)
    # Reference
    ref_val = promedios.get("Reference", 0.0)

    logger.info(f"PBS (fondo): {pbs_val:.2f}, Reference: {ref_val:.2f}")

    ref_neta = max(EPSILON, ref_val - pbs_val)

    resultados = {}
    for nombre, intens_bruta in promedios.items():
        if nombre in ("Reference", "PBS"):
            continue

        neta = max(0.0, intens_bruta - pbs_val)
        normalizada = round(neta / ref_neta, 6)

        if nombre in PRO_APOPTOTICAS:
            tipo = "pro-apoptótica"
        elif nombre in ANTI_APOPTOTICAS:
            tipo = "anti-apoptótica"
        else:
            tipo = "otra"

        resultados[nombre] = {
            'intensidad_bruta': round(intens_bruta, 2),
            'intensidad_neta': round(neta, 2),
            'normalizada': normalizada,
            'tipo': tipo,
            'num_spots': len(grupos[nombre]),
        }

    return {
        'proteinas': resultados,
        'pbs': pbs_val,
        'referencia': ref_val,
        'referencia_neta': round(ref_neta, 2),
    }


# =====================================================================
# 5. Pipeline de Extracción de Datos
# =====================================================================

def analizar_una_imagen(ruta_imagen: str, num_strips_esperados: int = 4) -> Tuple[List[Dict], np.ndarray, List[np.ndarray]]:
    """
    Carga la imagen única, detecta todos los spots para encontrar 
    la zona real de las membranas, y divide ESA zona uniformemente 
    en el número de strips esperados.
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar: {ruta_imagen}")

    logger.info(f"Imagen cargada: {imagen.shape[1]}x{imagen.shape[0]}")

    # 1. Encontrar todos los spots en la imagen gigante
    spots_globales = detectar_spots_en_strip(imagen)
    
    if not spots_globales:
        raise ValueError("No se encontraron spots en la imagen. Verifique la calidad de la imagen.")

    # 2. Encontrar el inicio y fin REAL de las tirillas (ignorando márgenes blancos enormes)
    xs = [s['centro'][0] for s in spots_globales]
    x_min, x_max = min(xs), max(xs)
    
    ancho_total_spots = x_max - x_min
    logger.info(f"Zona con spots detectada entre X:{x_min} y X:{x_max} (Ancho: {ancho_total_spots}px)")
    
    if ancho_total_spots < 50:
        raise ValueError("Todos los spots están demasiado juntos. Imposible dividir en strips.")

    # 3. Dividir esa zona uniformemente
    ancho_por_strip = ancho_total_spots / num_strips_esperados
    
    cuantificaciones = []
    strips_imgs = []
    alto = imagen.shape[0]
    
    for i in range(num_strips_esperados):
        # Calcular límites matemáticos para este strip
        x1_ideal = x_min + (i * ancho_por_strip)
        x2_ideal = x_min + ((i + 1) * ancho_por_strip)
        
        # Añadir un margen de seguridad pequeño para no cortar spots en el borde
        margen = 10
        x1 = max(0, int(x1_ideal - margen))
        x2 = min(imagen.shape[1], int(x2_ideal + margen))
        
        strip_img = imagen[:, x1:x2].copy()
        strips_imgs.append(strip_img)
        logger.info(f"Strip {i+1} aislado en coordenadas X:[{x1}-{x2}]")
        
        # 4. Volver a detectar spots localmente para coordenadas correctas
        spots_strip = detectar_spots_en_strip(strip_img)
        
        # 5. Mapear al grid (A1-E24)
        spots_mapeados = mapear_spots_a_grid(spots_strip)
        
        # 6. Cuantificar
        cuant = cuantificar_proteinas(spots_mapeados)
        cuant['_spots_mapeados'] = spots_mapeados 
        cuantificaciones.append(cuant)
        
    logger.info(f"Se cuantificaron exitosamente {num_strips_esperados} strips.")
    return cuantificaciones, imagen, strips_imgs


# =====================================================================
# 6. Fold Change
# =====================================================================

def calcular_fold_change(cuant_control: Dict, cuant_tratamiento: Dict) -> Dict:
    prot_ctrl = cuant_control['proteinas']
    prot_trat = cuant_tratamiento['proteinas']
    todas = set(prot_ctrl.keys()) | set(prot_trat.keys())

    resultados = {}
    for nombre in sorted(todas):
        ctrl = prot_ctrl.get(nombre, {})
        trat = prot_trat.get(nombre, {})
        nc = ctrl.get('normalizada', 0)
        nt = trat.get('normalizada', 0)
        tipo = ctrl.get('tipo', trat.get('tipo', 'otra'))

        if nc < 0.001 and nt < 0.001:
            continue

        if nc < 0.001:
            fold = float('inf')
            estado = "▲ APARECIÓ"
        elif nt < 0.001:
            fold = 0.0
            estado = "▼ DESAPARECIÓ"
        else:
            fold = round(nt / nc, 4)
            if fold > 1.5:
                estado = "▲ AUMENTÓ"
            elif fold < 0.67:
                estado = "▼ DISMINUYÓ"
            else:
                estado = "─ SIN CAMBIO"

        resultados[nombre] = {
            'ctrl_bruta': ctrl.get('intensidad_bruta', 0),
            'trat_bruta': trat.get('intensidad_bruta', 0),
            'ctrl_norm': nc,
            'trat_norm': nt,
            'fold_change': fold,
            'estado': estado,
            'tipo': tipo,
        }

    return resultados


# =====================================================================
# 6.b. Promedio de réplicas internas (strips 1-2 vs 3-4 del kit ARY009)
# =====================================================================

def promediar_cuantificaciones(cuants: List[Dict]) -> Dict:
    """
    Promedia varias cuantificaciones (típicamente 2 réplicas internas del kit).
    Devuelve la misma estructura de cuantificar() pero con valores promediados
    y agregados de desviación estándar para cada proteína.
    """
    if not cuants:
        return {'proteinas': {}, 'pbs': 0.0, 'referencia': 0.0, 'referencia_neta': 0.0}
    if len(cuants) == 1:
        # Caso degenerado: una sola réplica
        c = dict(cuants[0])
        c['proteinas'] = {p: {**d, 'std_norm': 0.0} for p, d in cuants[0]['proteinas'].items()}
        return c

    # Recolectar nombres de todas las proteínas vistas en cualquier réplica
    todas = set()
    for c in cuants:
        todas |= set(c['proteinas'].keys())

    proteinas_avg: Dict[str, Dict] = {}
    for nombre in todas:
        brutas, netas, normas, tipos = [], [], [], []
        for c in cuants:
            d = c['proteinas'].get(nombre)
            if not d:
                continue
            brutas.append(d.get('intensidad_bruta', 0.0))
            netas.append(d.get('intensidad_neta', 0.0))
            normas.append(d.get('normalizada', 0.0))
            tipos.append(d.get('tipo', 'otra'))
        if not normas:
            continue
        proteinas_avg[nombre] = {
            'intensidad_bruta': round(float(np.mean(brutas)), 2),
            'intensidad_neta':  round(float(np.mean(netas)), 2),
            'normalizada':      round(float(np.mean(normas)), 6),
            'std_norm':         round(float(np.std(normas, ddof=0)), 6),
            'n_replicas':       len(normas),
            'tipo':             tipos[0],
        }

    # Promediar también PBS y Reference para reportar
    pbs  = float(np.mean([c.get('pbs', 0.0) for c in cuants]))
    refn = float(np.mean([c.get('referencia_neta', 0.0) for c in cuants]))
    ref  = float(np.mean([c.get('referencia', 0.0) for c in cuants]))

    return {
        'proteinas': proteinas_avg,
        'pbs': round(pbs, 2),
        'referencia': round(ref, 2),
        'referencia_neta': round(refn, 2),
        'n_replicas': len(cuants),
    }


def calcular_fold_change_con_error(
    cuant_control_avg: Dict,
    cuant_trat_avg: Dict,
) -> Dict:
    """
    Calcula Fold Change usando promedios de réplicas internas y propaga
    el error (desviación estándar) usando reglas de propagación de incertidumbres.

    Para FC = T / C, el error relativo se propaga como:
        sigma_FC / FC = sqrt( (sigma_T / T)² + (sigma_C / C)² )
    """
    prot_c = cuant_control_avg['proteinas']
    prot_t = cuant_trat_avg['proteinas']
    todas = set(prot_c.keys()) | set(prot_t.keys())

    resultados = {}
    for nombre in sorted(todas):
        c = prot_c.get(nombre, {})
        t = prot_t.get(nombre, {})
        nc = c.get('normalizada', 0.0)
        nt = t.get('normalizada', 0.0)
        sc = c.get('std_norm', 0.0)
        st = t.get('std_norm', 0.0)
        tipo = c.get('tipo', t.get('tipo', 'otra'))

        if nc < 0.001 and nt < 0.001:
            continue

        if nc < 0.001:
            fold = float('inf')
            err = 0.0
            estado = "▲ APARECIÓ"
        elif nt < 0.001:
            fold = 0.0
            err = 0.0
            estado = "▼ DESAPARECIÓ"
        else:
            fold = round(nt / nc, 4)
            # Propagación de error relativo
            try:
                rel = ((st / nt) ** 2 + (sc / nc) ** 2) ** 0.5
                err = round(fold * rel, 4)
            except Exception:
                err = 0.0
            if fold > 1.5:
                estado = "▲ AUMENTÓ"
            elif fold < 0.67:
                estado = "▼ DISMINUYÓ"
            else:
                estado = "─ SIN CAMBIO"

        resultados[nombre] = {
            'ctrl_bruta': c.get('intensidad_bruta', 0.0),
            'trat_bruta': t.get('intensidad_bruta', 0.0),
            'ctrl_norm': nc,
            'trat_norm': nt,
            'ctrl_std': sc,
            'trat_std': st,
            'fold_change': fold,
            'fold_err': err,
            'estado': estado,
            'tipo': tipo,
            'n_ctrl': c.get('n_replicas', 1),
            'n_trat': t.get('n_replicas', 1),
        }

    return resultados


# =====================================================================
# 7. Dibujar spots etiquetados
# =====================================================================

def dibujar_spots_en_strip(strip_bgr: np.ndarray,
                             spots_mapeados: List[Dict]) -> np.ndarray:
    """
    Dibuja los spots detectados sobre la sub-imagen del strip.
    Refs en rojo, PBS en azul, muestras en verde.
    """
    img = strip_bgr.copy()

    for spot in spots_mapeados:
        x, y, w, h = spot['bbox']
        nombre = spot.get('proteina', '?')
        coord = spot.get('coordenada', '?')

        if nombre == "Reference":
            color = (0, 0, 255)
            grosor = 2
        elif nombre == "PBS":
            color = (255, 100, 0)
            grosor = 2
        else:
            color = (0, 255, 0)
            grosor = 1

        cv2.rectangle(img, (x, y), (x + w, y + h), color, grosor)

        # Etiqueta
        escala = 0.35
        etiqueta = coord
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), (0, 0, 0), -1)
        cv2.putText(img, etiqueta, (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1)

    return img


# =====================================================================
# 8. Analizador basado en grilla fija (Reference Spot anchoring)
# =====================================================================

class AnalizadorGrillaARY009:
    """
    Alternativa robusta a mapear_spots_a_grid.

    En lugar de detectar spots libres y luego inferir la grilla,
    usa los Reference Spots de esquina como anclas geométricas y
    muestrea intensidad directamente en las posiciones conocidas del ARY009.

    Ventajas:
    - Funciona aunque spots débiles no sean detectados por threshold
    - No depende del número de spots encontrados para calcular la escala
    - Maneja perspectiva/rotación mediante interpolación bilineal
    """

    # Posición X relativa de cada columna (0.0 = A, 1.0 = E).
    # Según las dimensiones físicas del kit ARY009 (datasheet R&D Systems):
    #   A-B, B-C, C-D ≈ 4.5 mm cada uno (espaciado uniforme)
    #   D-E ≈ 7.5 mm (gap mayor porque E es solo columna de referencia)
    #   Total A-E ≈ 21 mm → fracciones: B=4.5/21, C=9/21, D=13.5/21
    # Geometria del kit, definida una sola vez en mapa_array.T_COLS
    # y compartida con gridfit_ary009 para que no puedan divergir.
    _T_COLS = T_COLS
    # Centroide de los pares de referencia en coordenadas de fila (1-24)
    _FILA_REF_TOP = 1.5    # centro de rows 1 y 2
    _FILA_REF_BOT = 23.5   # centro de rows 23 y 24

    def __init__(self, radio_muestreo: int = 8, modo_rigido: bool = False):
        """
        Args:
            radio_muestreo: radio de la caja de muestreo en píxeles.
            modo_rigido: si True, omite snap/ICP/refinamiento. Las posiciones
                         se calculan solo por interpolación geométrica desde
                         las 3 referencias. Útil para máxima reproducibilidad
                         cuando las membranas tienen layout muy uniforme.
        """
        self.radio = radio_muestreo
        self.modo_rigido = modo_rigido
        self._strips_visual = []

    def get_radio_dinamico(self, tl: Tuple[float, float], bl: Tuple[float, float]) -> int:
        """
        Calcula un radio de muestreo adaptativo basado en la distancia real entre filas.
        Garantiza que los cuadros no se solapen verticalmente.
        """
        dist_y = abs(bl[1] - tl[1])
        filas_diff = self._FILA_REF_BOT - self._FILA_REF_TOP
        if filas_diff <= 0:
            return self.radio
        espacio_fila = dist_y / filas_diff
        # El radio es la mitad del ancho de la caja. 
        # Usamos 40% del espacio de la fila para dejar un margen del 20% entre cajas.
        r = int(espacio_fila * 0.40)
        return max(2, r)

    # ------------------------------------------------------------------
    def separar_strips(
        self,
        imagen_gris: np.ndarray,
        n: int = 4,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detecta N tiras verticales en la imagen y devuelve recortes.
        Returns: [(roi_gris, (x0, y0, ancho, alto)), ...]

        Estrategia dual:
        1. Primero intenta por contraste de brillo (membrana gris vs fondo blanco).
           Es más robusto porque la membrana completa es visible, no solo los spots.
        2. Si falla, usa Black-Hat para aislar manchas oscuras (spots únicamente).
        """
        h, w = imagen_gris.shape

        # ── ESTRATEGIA 1: Detección por brillo de la membrana ──
        # La membrana es ligeramente más oscura que el fondo blanco puro.
        # Esto captura la SILUETA completa de cada tirilla, no solo los spots.
        max_val = np.iinfo(imagen_gris.dtype).max if imagen_gris.dtype.kind == 'u' else 255
        fondo_claro = np.mean(imagen_gris) > (max_val * 0.35)

        resultado_brillo = None
        if fondo_claro:
            resultado_brillo = self._separar_por_brillo(imagen_gris, n)

        # _segmentar_proyeccion localiza siempre n segmentos (o devuelve None),
        # pero descarta los que no tienen ni un pixel de membrana. Ese descarte
        # NO es un fallo de la estrategia: exigir len == n aqui tiraria a la
        # basura las tirillas buenas que si se localizaron en esa membrana y
        # caeria al Black-Hat (o a []) por un hueco que es real.
        if resultado_brillo:
            logger.info("Separación por contraste de brillo de la membrana.")
            return resultado_brillo

        # ── ESTRATEGIA 2: Detección por Black-Hat (spots oscuros) ──
        resultado_bhat = self._separar_por_blackhat(imagen_gris, n, fondo_claro)
        if resultado_bhat:
            return resultado_bhat

        logger.warning("No se pudieron separar los strips.")
        return []

    # ------------------------------------------------------------------
    def _separar_por_brillo(
        self, imagen_gris: np.ndarray, n: int,
    ) -> Optional[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        """
        Detecta strips por el contraste entre la membrana (gris) y el fondo (blanco).
        Usa Otsu + cierre morfológico para obtener la silueta de cada tirilla.
        """
        h, w = imagen_gris.shape

        # Suavizado fuerte para eliminar los spots individuales y dejar solo la silueta
        blur = cv2.GaussianBlur(imagen_gris, (15, 15), 0)

        # Otsu separa el fondo blanco de la membrana gris
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Cierre morfológico agresivo para unir toda la membrana como un bloque
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, h // 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        # Apertura para eliminar ruido fino
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        # Proyección vertical (cuántos píxeles de membrana hay en cada columna X)
        proj = mask.astype(np.float64).sum(axis=0)
        if proj.max() < h * 0.05:
            return None

        # Suavizado de la proyección
        smooth = max(5, w // (n * 5))
        proj_s = np.convolve(proj, np.ones(smooth) / smooth, mode='same')

        return self._segmentar_proyeccion(imagen_gris, proj_s, mask, n)

    # ------------------------------------------------------------------
    def _separar_por_blackhat(
        self, imagen_gris: np.ndarray, n: int, fondo_claro: bool,
    ) -> Optional[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        """
        Detecta strips usando solo la señal de los spots oscuros (Black-Hat).
        """
        h, w = imagen_gris.shape

        tam_k = 51
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_k, tam_k))
        if fondo_claro:
            bhat = cv2.morphologyEx(imagen_gris, cv2.MORPH_BLACKHAT, kernel_bh)
        else:
            bhat = cv2.morphologyEx(imagen_gris, cv2.MORPH_TOPHAT, kernel_bh)

        _, bin_bhat = cv2.threshold(bhat, 8, 255, cv2.THRESH_BINARY)

        proj_bhat = bin_bhat.astype(np.float64).sum(axis=0)
        smooth = max(5, w // (n * 4))
        proj_s = np.convolve(proj_bhat, np.ones(smooth) / smooth, mode='same')

        if proj_s.max() < 1:
            logger.warning("Black-Hat no detectó manchas.")
            return None

        return self._segmentar_proyeccion(imagen_gris, proj_s, bin_bhat, n)

    # ------------------------------------------------------------------
    @staticmethod
    def _detectar_n_picos(
        proj: np.ndarray, n: int, x_min: int, x_max: int,
        perfil_membrana: Optional[np.ndarray] = None,
    ) -> Optional[List[int]]:
        """
        Detecta N picos (máximos locales) en la proyección, dentro del rango
        [x_min, x_max]. Robusto a variaciones de altura: si una tirilla tiene
        señal débil pero genera un máximo local distinto, será detectado.

        perfil_membrana: pixeles de membrana por columna (suma vertical de la
        mascara). Si se pasa, un candidato que no tenga membrana en su entorno
        se descarta como ruido y la busqueda SIGUE por el candidato siguiente.

        Returns: lista de N posiciones X de los picos, o None si no encuentra
                 exactamente N picos plausibles.
        """
        region = proj[x_min:x_max + 1].copy()
        if len(region) < n * 8:
            return None

        # Suavizado adicional para evitar picos espurios por ruido
        ventana_smooth = max(3, len(region) // (n * 12))
        if ventana_smooth % 2 == 0:
            ventana_smooth += 1
        kernel = np.ones(ventana_smooth) / ventana_smooth
        region_s = np.convolve(region, kernel, mode='same')

        # Distancia mínima entre picos: 1/(n+1) del rango (para evitar picos
        # demasiado cercanos que sean uno solo)
        min_dist = len(region) // (n + 1)
        win_half = max(2, min_dist // 2)

        # Encontrar todos los máximos locales
        candidatos = []
        for i in range(1, len(region_s) - 1):
            ventana = region_s[max(0, i - win_half):i + win_half + 1]
            # Es máximo en la ventana?
            if region_s[i] >= ventana.max() - 1e-9:
                candidatos.append((i, region_s[i]))

        if len(candidatos) < n:
            return None

        candidatos.sort(key=lambda c: -c[1])

        # Seleccion de picos: se recorren los candidatos de mas alto a mas bajo
        # y se acepta uno solo si (a) tiene membrana debajo y (b) esta separado
        # de los ya aceptados. El que no cumple se SALTA y la busqueda sigue.
        #
        # Modo de falla que evita: antes se tomaban los n candidatos mas altos
        # y si quedaban demasiado juntos se devolvia None. Dos defectos de eso:
        #   1. Los maximos locales de una meseta son todos candidatos con la
        #      MISMA altura, asi que el top-n se llenaba con puntos vecinos de
        #      una sola membrana (medido: C.mama top-4 = 584,585,586,587) y la
        #      funcion devolvia None sobre la proyeccion de la mascara, que es
        #      justo la que esta bien alineada con las membranas.
        #   2. Al devolver None se caia al fallback de la proyeccion DIRECTA,
        #      cuyo rango de contenido abarca toda la foto porque el ruido de
        #      escaneo del borde pasa el umbral. Eso infla min_dist (207 px
        #      contra un paso real entre membranas de ~128) y la ventana de
        #      maximos se come membranas reales, ademas de aceptar picos sobre
        #      el borde vacio (medido: x=101 en C.mama, x=36 en Lineas, x=1019
        #      en agraz, los tres con 0 pixeles de membrana). Un pico falso
        #      corre TODOS los limites porque los segmentos se cortan a media
        #      distancia entre picos: sobre las 6 fotos de prueba eso dejaba 3
        #      segmentos en vez de 4 (uno caia en el vacio y lo descartaba el
        #      guardian de abajo) y los vecinos a caballo entre dos membranas
        #      (Lineas M1 ratio 1.741, C.mama M3 1.568).
        #
        # El piso es 5 % del maximo del perfil: la banda de membrana mas debil
        # de las 6 fotos llega al 8.1 % (C.mama M2 y M4, 18 de 223 px) y todo
        # lo que cae fuera de las membranas marca exactamente 0 % en las 6.
        # No es un umbral de aceptacion de tirillas: es la exigencia de que un
        # "pico" tenga membrana fisica debajo antes de mover ningun limite.
        piso_cont = 0.0
        if perfil_membrana is not None and perfil_membrana.size:
            piso_cont = float(perfil_membrana.max()) * 0.05
        # Entorno de ~1/3 del paso entre tirillas: el ancho de una membrana,
        # no el del pasillo entre dos, para que el pico tenga que estar SOBRE
        # la banda y no al lado de ella.
        radio_cont = max(2, min_dist // 3)
        sep_min = max(1, min_dist // 2)

        seleccionados: List[int] = []
        for idx, _altura in candidatos:
            x_abs = x_min + idx
            if piso_cont > 0:
                lo = max(0, x_abs - radio_cont)
                hi = min(int(perfil_membrana.size), x_abs + radio_cont + 1)
                if hi <= lo or float(perfil_membrana[lo:hi].max()) < piso_cont:
                    continue  # ruido de borde: no hay membrana, no es tirilla
            if any(abs(x_abs - s) < sep_min for s in seleccionados):
                continue  # mismo pico que uno ya aceptado (meseta o vecino)
            seleccionados.append(x_abs)
            if len(seleccionados) == n:
                break

        if len(seleccionados) != n:
            return None

        # Ya estan en coordenadas absolutas
        return sorted(seleccionados)

    # ------------------------------------------------------------------
    def _segmentar_proyeccion(
        self,
        imagen_gris: np.ndarray,
        proj_s: np.ndarray,
        mask_vertical: np.ndarray,
        n: int,
    ) -> Optional[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        """
        A partir de una proyección horizontal suavizada, segmenta N strips.
        Intenta detectar valles entre strips; si falla, divide uniformemente.
        """
        h, w = imagen_gris.shape

        proj_norm = proj_s - proj_s.min()
        if proj_norm.max() < 1:
            return None

        # Límites globales de contenido
        thresh_global = proj_norm.max() * 0.05
        xs_content = np.where(proj_norm > thresh_global)[0]
        if xs_content.size == 0:
            return None
        x_min, x_max = int(xs_content[0]), int(xs_content[-1])
        span = max(1, x_max - x_min)

        # Detectar segmentos sobre el umbral
        thresh_seg = proj_norm.max() * 0.15
        tiene = proj_norm > thresh_seg
        segs: List[Tuple[int, int]] = []
        en, ini = False, 0
        for i, v in enumerate(tiene):
            if v and not en:
                ini = i; en = True
            elif not v and en:
                segs.append((ini, i)); en = False
        if en:
            segs.append((ini, w))

        # Fusionar segmentos cercanos (huecos dentro de un mismo strip)
        gap_max = span // (n * 2)
        merged: List[List[int]] = []
        for s in sorted(segs, key=lambda x: x[0]):
            if merged and s[0] - merged[-1][1] < gap_max:
                merged[-1][1] = s[1]
            else:
                merged.append([s[0], s[1]])

        ancho_min = span // (n * 4)
        merged = [s for s in merged if s[1] - s[0] > ancho_min]

        es_fallback_uniforme = False

        # ── Intento PRIMARIO: detectar N PICOS (máximos locales) ──
        # Probamos primero con la proyección suavizada (que ya viene del
        # llamador). Si no encuentra N picos, intentamos con la proyección
        # directa de la imagen invertida (sin morfología agresiva) que
        # preserva mejor las tirillas débiles.
        # Evidencia fisica de membrana por columna. Es la MISMA mascara con la
        # que mas abajo se descarta un segmento sin pixeles de membrana: asi el
        # pico que se acepta aqui cae donde ese guardian no va a tirar la caja.
        perfil_membrana = (mask_vertical > 0).sum(axis=0).astype(np.float64)

        picos_detectados = self._detectar_n_picos(proj_norm, n, x_min, x_max,
                                                  perfil_membrana)
        # Rango de contenido de la proyeccion de la que salieron los picos que
        # finalmente ganan. Se guarda aparte porque el reintento de abajo usa
        # OTRA proyeccion, con su propio rango: recortar los picos de una
        # proyeccion con los limites de la otra produce cajas invertidas.
        rango_picos = (x_min, x_max)

        if picos_detectados is None or len(picos_detectados) != n:
            # Fallback: proyección sobre la imagen invertida DIRECTA
            # (sin closing/opening que pueda eliminar tirillas tenues)
            inv = 255 - imagen_gris
            proj_directa = inv.astype(np.float64).sum(axis=0)
            # Suavizado proporcional
            sm = max(5, len(proj_directa) // (n * 5))
            proj_dir_s = np.convolve(proj_directa, np.ones(sm) / sm, mode='same')
            proj_dir_norm = proj_dir_s - proj_dir_s.min()

            # Buscar contenido (umbral más bajo: 3%)
            if proj_dir_norm.max() > 1:
                xs_cont2 = np.where(proj_dir_norm > proj_dir_norm.max() * 0.03)[0]
                if xs_cont2.size > 0:
                    xm2, xM2 = int(xs_cont2[0]), int(xs_cont2[-1])
                    picos_alt = self._detectar_n_picos(proj_dir_norm, n,
                                                       xm2, xM2,
                                                       perfil_membrana)
                    if picos_alt is not None and len(picos_alt) == n:
                        picos_detectados = picos_alt
                        # Los limites tienen que viajar junto con los picos:
                        # la proyeccion directa arranca mas afuera que la
                        # suavizada, y usar x_min/x_max aqui empujaba el borde
                        # de la primera tirilla por delante de su propio final
                        # (o el de la ultima por detras de su inicio), dejando
                        # un recorte de ancho negativo -> tirilla vacia.
                        rango_picos = (xm2, xM2)
                        logger.info("Picos detectados sobre proyección directa (más sensible).")
        if picos_detectados is not None and len(picos_detectados) == n:
            # Construir segmentos centrados en los picos
            # Cada segmento se extiende a mitad de camino al pico vecino
            segs_picos = []
            picos = sorted(picos_detectados)
            px_min, px_max = rango_picos
            for i, pico in enumerate(picos):
                if i == 0:
                    # primer strip: desde inicio hasta media distancia al siguiente
                    x0 = max(px_min, pico - (picos[1] - pico) // 2) if len(picos) > 1 else px_min
                else:
                    x0 = (picos[i - 1] + pico) // 2
                if i == len(picos) - 1:
                    x1 = min(px_max, pico + (pico - picos[-2]) // 2) if len(picos) > 1 else px_max
                else:
                    x1 = (pico + picos[i + 1]) // 2
                segs_picos.append((x0, x1))
            final_segs = segs_picos
            logger.info(f"Segmentación por PICOS: {n} picos en {picos}")
        elif len(merged) == n:
            final_segs = [(s[0], s[1]) for s in merged]
        else:
            # ── Intento intermedio: detectar VALLES entre tirillas ──
            # Cuando las tirillas se mergen en 1 segmento, buscamos los N-1
            # valles más profundos en la proyección dentro del rango [x_min, x_max].
            # Cada valle separa una tirilla de la siguiente.
            valles_detectados = None
            if len(merged) == 1:
                # Trabajar en el rango con contenido
                region = proj_norm[x_min:x_max + 1].copy()
                if len(region) > n * 10:
                    # Suavizar adicional para encontrar valles robustos
                    smooth_v = max(3, len(region) // (n * 8))
                    region_s = np.convolve(region, np.ones(smooth_v) / smooth_v, mode='same')

                    # Buscar N-1 valles (mínimos locales)
                    # Usando una ventana mínima de 1/n del ancho para evitar valles falsos
                    min_dist = len(region) // (n + 1)
                    # Generar candidatos: invertir la señal y buscar máximos
                    inv_sig = -region_s
                    candidatos_idx = []
                    for i in range(min_dist, len(inv_sig) - min_dist):
                        # Es mínimo local en una ventana de min_dist/2?
                        win_half = min_dist // 2
                        ventana = inv_sig[max(0, i - win_half):i + win_half + 1]
                        if inv_sig[i] >= ventana.max() - 1e-9:
                            candidatos_idx.append((i, inv_sig[i]))

                    # Tomar los N-1 mejores valles (los más profundos)
                    candidatos_idx.sort(key=lambda x: -x[1])
                    valles_seleccionados = sorted([x[0] for x in candidatos_idx[:n - 1]])

                    if len(valles_seleccionados) == n - 1:
                        # Construir segmentos a partir de los valles
                        valles_seleccionados = [x_min] + [x_min + v for v in valles_seleccionados] + [x_max]
                        valles_detectados = []
                        for i in range(n):
                            valles_detectados.append((valles_seleccionados[i], valles_seleccionados[i + 1]))

            if valles_detectados is not None:
                final_segs = valles_detectados
                logger.info(f"Segmentación por VALLES detectados ({n - 1} valles).")
            else:
                # Fallback final: división uniforme con margen extendido
                margen_extra = int(span * 0.12)
                x_lo = max(0, x_min - margen_extra)
                x_hi = min(w, x_max + margen_extra)
                span_ext = x_hi - x_lo
                strip_w = span_ext / n
                final_segs = [
                    (int(x_lo + i * strip_w), int(x_lo + (i + 1) * strip_w))
                    for i in range(n)
                ]
                es_fallback_uniforme = True
                logger.info(f"Fallback uniforme ({len(merged)} seg → {n} strips, span ext={span_ext}).")

        # Calcular margen lateral SIN producir solapamiento entre strips vecinos.
        if es_fallback_uniforme:
            # Sin margen extra: los segmentos ya cubren todo el rango extendido.
            mg_x = 0
        else:
            # En detección por valles: margen pequeño limitado por gap mínimo
            gaps = []
            for i in range(len(final_segs) - 1):
                gaps.append(final_segs[i + 1][0] - final_segs[i][1])
            min_gap = min(gaps) if gaps else span // n
            mg_x = max(0, min(span // (n * 10), min_gap // 3))

        strips = []
        for i_seg, (x0, x1) in enumerate(final_segs):
            x0 = max(0, x0 - mg_x)
            x1 = min(w, x1 + mg_x)

            col_mask = mask_vertical[:, x0:x1]

            # Recorte en X al contenido real de la mascara: el equivalente del
            # recorte en Y que se hace unas lineas mas abajo con proj_y/rows.
            # Se aplica a LOS DOS BORDES DE TODAS las tirillas.
            # Modo de falla que evita: ningun borde de segmento conoce el ancho
            # real de la membrana que encierra. El exterior sale de
            # "pico -/+ media distancia al vecino" clampeado contra px_min/px_max,
            # y el interior es el punto medio entre dos picos, que reparte 50/50
            # el pasillo vacio entre dos membranas: las dos cajas se quedan con
            # media calle de fondo. En ambos casos la caja se estira sobre
            # membrana vacia, y el ajuste de grilla ancla las referencias una
            # columna corrida y varias filas por encima del par real.
            # Medido: en SW480-Aspirina M1 el borde izquierdo (exterior) sobraba
            # 33 px y row_spacing caia de 9.9 a 7.7 px (relacion de celda
            # 1.335 -> 1.698, grilla comprimida 22 % y 5 filas de spots sin
            # cubrir, incluido el A23/A24 verdadero, con el PBS cayendo sobre
            # senal real). El mismo defecto en bordes INTERIORES mantenia
            # SW480-Aspirina M2 en ratio 1.701 y Sw480-agraz M3 en 1.714 -- dos
            # tirillas entregando 37 proteinas sobre grilla mal puesta -- y
            # recortarlos las baja a 1.390 y 1.391. Por eso no alcanza con los
            # bordes exteriores: la cuenta de tirillas SANAS sube al recortar
            # los cuatro lados.
            # SOLO ENCOGE: si el ajuste quisiera ampliar la caja se deja como
            # esta, porque ampliar es justamente el defecto.
            proj_x = (col_mask > 0).sum(axis=0)
            cols_c = np.where(proj_x > 0)[0]
            if cols_c.size:
                # Mismo margen que el recorte en Y de abajo (max(15, 6 %)):
                # la mascara marca spots, no la membrana entera, asi que
                # el contenido termina en el borde del spot de la columna
                # exterior y hace falta dejarle el reborde de membrana.
                # Con menos de ~12 px el recorte empezaria a morder el
                # borde derecho de las dos membranas que hoy van 4/4.
                mg_xc = max(15, int((cols_c[-1] - cols_c[0]) * 0.06))
                nx0 = max(x0, x0 + int(cols_c[0]) - mg_xc)
                nx1 = min(x1, x0 + int(cols_c[-1]) + 1 + mg_xc)
                if (nx0, nx1) != (x0, x1) and nx1 - nx0 >= 20:
                    logger.info(
                        f"Strip {i_seg+1}: caja encogida al "
                        f"contenido x=[{x0},{x1}] -> x=[{nx0},{nx1}]."
                    )
                    x0, x1 = nx0, nx1
                    col_mask = mask_vertical[:, x0:x1]

            proj_y = col_mask.astype(np.float64).sum(axis=1)
            rows = np.where(proj_y > 0)[0]
            if rows.size == 0:
                # En este rango de x la mascara no tiene NI UN pixel de
                # membrana: aqui no hay tirilla que recortar. Recortar la foto
                # completa (y0,y1 = 0,h) fabricaba una "tirilla" de ruido de
                # escaneo -- 0.00% de pixeles oscuros, contra 0.8-6.2% en una
                # membrana real -- que encontrar_referencias llegaba a aceptar
                # y entregaba 37 proteinas inventadas sobre el fondo. Sin
                # membrana no hay dato: el segmento se descarta y la lista sale
                # con menos de n (separar_strips y analizar_imagen lo toleran).
                logger.warning(
                    f"Segmento x=[{x0},{x1}]: sin pixeles de membrana en la "
                    f"mascara, se descarta (no se recorta la imagen completa)."
                )
                continue

            mg_y = max(15, int((rows[-1] - rows[0]) * 0.06))
            y0 = max(0, int(rows[0]) - mg_y)
            y1 = min(h, int(rows[-1]) + mg_y)

            strips.append((imagen_gris[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)))

        logger.info(f"Strips separados: {len(strips)}/{n}")
        return strips

    # ------------------------------------------------------------------
    def encontrar_referencias(
        self,
        strip_gris: np.ndarray,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        Encuentra los 3 pares de Reference Spots del kit ARY009 y devuelve
        anchors compatibles con _pos_pixel():
            tl = posición del par A_top    (col A, filas 1-2)
            tr = posición del par E_top    (col E, filas 1-2)
            bl = posición del par A_bottom (col A, filas 23-24)

        Estrategias en orden de prioridad. Cada candidato se VALIDA con un
        grid-fit test (al menos 25% de las posiciones ARY009 deben caer
        sobre spots reales detectados). El primer candidato válido GANA.

          1. Bandas top/bottom
          2. Pares verticales
          3. Ajuste por grid (RANSAC)
          4. Legacy (último recurso)
        """
        # Validar entrada
        if strip_gris is None or strip_gris.size == 0:
            return None
        if strip_gris.shape[0] < 30 or strip_gris.shape[1] < 20:
            logger.warning(f"Strip muy pequeño ({strip_gris.shape}), saltando.")
            return None

        # Detectar todos los spots una sola vez (para validación)
        spots_todos = self._detectar_todos_los_spots(strip_gris)
        spots_xy = (np.array([[s[0], s[1]] for s in spots_todos], dtype=np.float32)
                    if spots_todos else np.empty((0, 2), dtype=np.float32))
        # Se guardan para que _asignar_anchors pueda centrar cada ancla en su
        # PAR de duplicados. Las tres estrategias desembocan ahí y ninguna
        # recibe la lista de spots, pero aquí ya está calculada una sola vez.
        self._spots_para_anclas = spots_todos

        h, w = strip_gris.shape

        def _aceptar(refs):
            """
            Validación adaptativa según calidad de las anclas:

            CASO 1 (anclas PERFECTAS): si TL, TR, BL están a < 5px de un spot
              real, el grid es confiable. Aceptar incluso con muestras débiles
              (pocos matches en columnas internas).

            CASO 2 (anclas APROXIMADAS): si TL, TR, BL están a 5-12px de un
              spot, validar con grid-fit en columnas.

            CASO 3 (anclas LEJOS): si alguna anchor está a > 12px de cualquier
              spot, las refs están mal detectadas. RECHAZAR.
            """
            if refs is None or spots_xy.size == 0:
                return False

            # Forma de la celda ANTES que nada: unos anclajes que describen un
            # grid imposible no sirven por muy bien que caigan sobre spots.
            # Va antes del CASO 1 a propósito — ese atajo aceptaba candidatos
            # con la geometría rota solo porque las 3 anclas tocaban un spot,
            # y así ganaba el primer candidato malo en vez de seguir probando
            # las estrategias siguientes, que suelen acertar.
            cs_cand, rs_cand, ratio_cand = self._ratio_celda(refs)
            if not (self.RATIO_CELDA_MIN <= ratio_cand <= self.RATIO_CELDA_MAX):
                logger.info(
                    f"  Val geom: celda {cs_cand:.1f}x{rs_cand:.1f}px "
                    f"(relación {ratio_cand:.2f}, esperada "
                    f"{self.RATIO_CELDA_MIN}-{self.RATIO_CELDA_MAX}) → RECHAZADO"
                )
                return False

            # Calcular distancia de cada anchor al spot más cercano
            radio_ancla = max(8, int(w * 0.12))
            dist_ancla_max = 0.0
            for nombre, (rx, ry) in zip(['TL', 'TR', 'BL'], refs):
                dx = spots_xy[:, 0] - rx
                dy = spots_xy[:, 1] - ry
                dist_min = float(((dx * dx + dy * dy)).min()) ** 0.5
                if dist_min > radio_ancla:
                    logger.info(
                        f"  Val anchor: {nombre}=({rx:.0f},{ry:.0f}) sin spot "
                        f"(d={dist_min:.0f}>{radio_ancla}) RECHAZADO"
                    )
                    return False
                dist_ancla_max = max(dist_ancla_max, dist_min)

            # COBERTURA: qué fracción de los spots detectados queda explicada
            # por alguna posición del grid. Es la prueba de que las anclas
            # describen ESTA membrana y no una desplazada que casualmente
            # apoya sus 3 esquinas sobre spots reales.
            cs_ref, rs_ref, _ = self._ratio_celda(refs)
            tol_cob_sq = (max(cs_ref, rs_ref) * 0.50) ** 2
            tl_c, tr_c, bl_c = refs
            fila_span_c = self._FILA_REF_BOT - self._FILA_REF_TOP
            grid_pts = []
            for (col_c, fila_c) in MAPA_PROTEINAS:
                t_col = self._T_COLS[col_c]
                t_row = (fila_c - self._FILA_REF_TOP) / fila_span_c
                x_A = tl_c[0] + t_row * (bl_c[0] - tl_c[0])
                grid_pts.append((x_A + t_col * (tr_c[0] - tl_c[0]),
                                 tl_c[1] + t_row * (bl_c[1] - tl_c[1])))
            grid_arr = np.array(grid_pts, dtype=np.float32)
            n_explicados = 0
            for sx_c, sy_c in spots_xy:
                ddx = grid_arr[:, 0] - sx_c
                ddy = grid_arr[:, 1] - sy_c
                if float((ddx * ddx + ddy * ddy).min()) <= tol_cob_sq:
                    n_explicados += 1
            cobertura = n_explicados / len(spots_xy)

            # CASO 1: anclas perfectas (todas a <5px de un spot real)
            # → aceptar sin más validación (muestra puede ser débil)
            if dist_ancla_max < 5.0:
                # El atajo existe para membranas tenues, donde hay tan pocos
                # spots que ninguna validación de grid significa nada. Pero si
                # hay spots de sobra, el grid tiene que explicarlos: unas anclas
                # que tocan 3 spots e ignoran otros 43 describen una grilla
                # desplazada, no una muestra débil.
                if len(spots_xy) < 15 or cobertura >= 0.35:
                    logger.info(
                        f"  Val OK: anclas perfectas (max dist={dist_ancla_max:.1f}px, "
                        f"cobertura {cobertura:.0%} de {len(spots_xy)} spots)"
                    )
                    return True
                logger.info(
                    f"  Val cobertura: anclas perfectas pero el grid solo explica "
                    f"{cobertura:.0%} de {len(spots_xy)} spots → RECHAZADO"
                )
                return False

            # 2. Calcular posiciones del grid y contar matches POR COLUMNA
            tl, tr, bl = refs
            sep_x = abs(tr[0] - tl[0])
            sep_y = abs(bl[1] - tl[1])
            if sep_x < 1 or sep_y < 1:
                return False
            row_spacing = sep_y / 22.0
            col_spacing = sep_x / 4.0
            tol_sq = (max(row_spacing, col_spacing) * 0.50) ** 2

            from collections import defaultdict
            matches_por_col = defaultdict(int)
            score = 0.0
            fila_span = self._FILA_REF_BOT - self._FILA_REF_TOP

            for (col, fila), proteina in MAPA_PROTEINAS.items():
                t_col = self._T_COLS[col]
                t_row = (fila - self._FILA_REF_TOP) / fila_span
                x_A = tl[0] + t_row * (bl[0] - tl[0])
                px = x_A + t_col * (tr[0] - tl[0])
                py = tl[1] + t_row * (bl[1] - tl[1])
                dx = spots_xy[:, 0] - px
                dy = spots_xy[:, 1] - py
                if float((dx * dx + dy * dy).min()) <= tol_sq:
                    matches_por_col[col] += 1
                    if proteina == 'Reference':
                        score += 5.0
                    elif proteina == 'PBS':
                        score += 2.0
                    else:
                        score += 1.0

            # Requisitos diferentes por columna según el contenido esperado en ARY009:
            #   E: solo 2 refs (E1, E2) → min 1 match
            #   A: 4 refs + algunas proteínas (depende muestra) → min 2 matches
            #   B, C: 24 proteínas (depende expresión) → min 2 matches
            #   D: 22 proteínas + 2 PBS → min 2 matches
            min_por_col = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1}
            # Pero al menos 3 columnas deben tener 3+ matches (grid bien distribuido)
            cols_con_3plus = sum(1 for c in 'ABCDE' if matches_por_col[c] >= 3)
            if cols_con_3plus < 3:
                cols_str = ",".join(f"{c}={matches_por_col[c]}" for c in 'ABCDE')
                logger.info(
                    f"  Val cols: {cols_str} → solo {cols_con_3plus} col(s) "
                    f"tienen 3+ matches → RECHAZADO"
                )
                return False
            cols_vacias = [c for c in 'ABCDE' if matches_por_col[c] < min_por_col[c]]
            if cols_vacias:
                cols_str = ",".join(f"{c}={matches_por_col[c]}" for c in 'ABCDE')
                logger.info(
                    f"  Val cols: {cols_str} → columnas {cols_vacias} "
                    f"con muy pocos matches → RECHAZADO"
                )
                return False

            # 3. Score global mínimo
            score_min = 18.0
            if score < score_min:
                logger.info(f"  Val score: {score:.0f}<{score_min:.0f} → RECHAZADO")
                return False

            cols_str = ",".join(f"{c}={matches_por_col[c]}" for c in 'ABCDE')
            logger.info(f"  Val OK: score={score:.0f} cols={cols_str}")
            return True

        # Estrategia 1: bandas top/bottom
        resultado = self._referencias_por_bandas(strip_gris)
        if _aceptar(resultado):
            return resultado

        # Estrategia 2: pares verticales
        logger.info("Probando detección por pares verticales.")
        resultado = self._referencias_por_pares(strip_gris)
        if _aceptar(resultado):
            return resultado

        # Estrategia 3: RANSAC grid-fitting (más caro pero busca el mejor encaje)
        logger.info("Probando ajuste por grid (RANSAC).")
        resultado = self._referencias_por_ajuste_grilla(strip_gris)
        if _aceptar(resultado):
            return resultado

        # Estrategia 4: legacy
        logger.info("Probando método legacy.")
        resultado = self._referencias_legacy(strip_gris)
        if _aceptar(resultado):
            return resultado

        # Estrategia 5 (último recurso): ajuste rígido guiado por TODOS los
        # spots, en vez de por 3 anclas. Las 4 estrategias anteriores buscan
        # primero los Reference Spots y derivan la grilla de ellos; cuando la
        # membrana está inclinada o corrida esa búsqueda falla y la tirilla se
        # perdía entera. Aquí se hace al revés: se ajusta la retícula completa
        # (traslación, escala, rotación — la topología nunca se deforma) contra
        # la nube de spots detectados, y las referencias se DEDUCEN del ajuste.
        # Va de últimas a propósito: donde las otras funcionan, no se toca nada.
        logger.info("Probando ajuste rígido guiado por spots (gridfit).")
        try:
            par, _info = gridfit_ary009.ajustar_ary009(
                spots_todos, strip_gris, rs_prior=strip_gris.shape[0] / 24.0,
            )
            if par is not None:
                resultado = gridfit_ary009.par_a_refs(par)
                if _aceptar(resultado):
                    logger.info("Referencias recuperadas por gridfit.")
                    return resultado
        except Exception as e:
            logger.warning(f"gridfit falló: {e}")

        # Ninguna estrategia produjo un grid-fit aceptable
        logger.warning("Ninguna estrategia produjo referencias con grid válido. "
                       "Usuario debe ajustar manualmente.")
        return None

    # ------------------------------------------------------------------
    def _referencias_por_ajuste_grilla(
        self, strip_gris: np.ndarray,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        Estrategia tipo RANSAC: detecta TODAS las manchas y prueba muchas
        combinaciones de anchors candidatas, eligiendo la que produce más
        coincidencias con las 120 posiciones de la grilla ARY009.

        Más robusto que buscar las 3 referencias directamente porque:
          - Usa la "constelación" completa de spots, no 3 puntos específicos
          - Si una referencia está faltante/débil, otras combinaciones la compensan
          - El mejor encaje aparece naturalmente cuando los anchors son correctos
        """
        h, w = strip_gris.shape

        # 1) Detectar TODOS los spots del strip (multi-escala)
        spots_todos = self._detectar_todos_los_spots(strip_gris)
        if len(spots_todos) < 8:
            logger.info(f"Ajuste grilla: muy pocos spots detectados ({len(spots_todos)})")
            return None

        # 2) Separar candidatos por zona (top 28% y bottom 28%)
        candidatos_top = [s for s in spots_todos if s[1] < h * 0.28]
        candidatos_bot = [s for s in spots_todos if s[1] > h * 0.72]

        if len(candidatos_top) < 2 or not candidatos_bot:
            logger.info("Ajuste grilla: sin candidatos en bandas top/bottom.")
            return None

        # Ordenar por área (más grandes primero) y limitar combinaciones
        candidatos_top.sort(key=lambda s: -s[2])
        candidatos_bot.sort(key=lambda s: -s[2])
        candidatos_top = candidatos_top[:12]
        candidatos_bot = candidatos_bot[:8]

        # 3) Array NumPy de todos los spots para evaluación rápida
        spots_xy = np.array([[s[0], s[1]] for s in spots_todos], dtype=np.float32)

        # 4) Probar todas las combinaciones válidas (TL, TR, BL)
        mejor_score = 0
        mejor_combo = None

        for ii in range(len(candidatos_top)):
            for jj in range(len(candidatos_top)):
                if ii == jj:
                    continue
                si, sj = candidatos_top[ii], candidatos_top[jj]
                # Probar como (left, right) — el orden importa para A vs E
                if si[0] >= sj[0]:
                    continue

                # Separación horizontal mínima
                sep_x = sj[0] - si[0]
                if sep_x < w * 0.30:
                    continue

                # Top spots deben estar a Y similar (misma fila ~1.5)
                if abs(si[1] - sj[1]) > h * 0.12:
                    continue

                for bl_c in candidatos_bot:
                    d_left = abs(bl_c[0] - si[0])
                    d_right = abs(bl_c[0] - sj[0])

                    # BL debe alinearse con uno de los dos (col A tiene bot ref)
                    if min(d_left, d_right) > w * 0.18:
                        continue

                    # Determinar cuál es A (alineado con BL) y cuál es E
                    if d_left < d_right:
                        # si es A (izquierda), sj es E (derecha)
                        tl_use = (float(si[0]), float(si[1]))
                        tr_use = (float(sj[0]), float(sj[1]))
                    else:
                        # sj es A (derecha), si es E (izquierda)
                        tl_use = (float(sj[0]), float(sj[1]))
                        tr_use = (float(si[0]), float(si[1]))
                    bl_use = (float(bl_c[0]), float(bl_c[1]))

                    # Evaluar esta hipótesis: ¿cuántas posiciones de la grilla
                    # tienen un spot detectado cerca?
                    score = self._evaluar_ajuste_grilla(tl_use, tr_use, bl_use, spots_xy)

                    if score > mejor_score:
                        mejor_score = score
                        mejor_combo = (tl_use, tr_use, bl_use)

        # Mínimo razonable: 25 posiciones encajan (de 120 totales)
        if mejor_combo is None or mejor_score < 25:
            logger.info(f"Ajuste grilla: mejor score insuficiente ({mejor_score})")
            return None

        logger.info(
            f"Referencias por AJUSTE GRILLA: score={mejor_score}  "
            f"TL={mejor_combo[0]}  TR={mejor_combo[1]}  BL={mejor_combo[2]}"
        )
        return mejor_combo

    # ------------------------------------------------------------------
    def _evaluar_ajuste_grilla(
        self,
        tl: Tuple[float, float],
        tr: Tuple[float, float],
        bl: Tuple[float, float],
        spots_xy: np.ndarray,
    ) -> float:
        """
        Score: para cada una de las 120 posiciones de la grilla ARY009,
        verifica si hay un spot detectado dentro de tolerancia.
        Referencias y PBS pesan más porque deben coincidir bien.
        """
        sep_x = abs(tr[0] - tl[0])
        sep_y = abs(bl[1] - tl[1])
        if sep_x < 1 or sep_y < 1:
            return 0

        row_spacing = sep_y / 22.0
        col_spacing = sep_x / 4.0
        tol_sq = (max(row_spacing, col_spacing) * 0.30) ** 2

        if spots_xy.size == 0:
            return 0

        score = 0.0
        fila_span = self._FILA_REF_BOT - self._FILA_REF_TOP

        for (col, fila), proteina in MAPA_PROTEINAS.items():
            t_col = self._T_COLS[col]
            t_row = (fila - self._FILA_REF_TOP) / fila_span

            # Interpolación bilineal
            x_A = tl[0] + t_row * (bl[0] - tl[0])
            px = x_A + t_col * (tr[0] - tl[0])
            py = tl[1] + t_row * (bl[1] - tl[1])

            # Distancia mínima al spot detectado más cercano
            dx = spots_xy[:, 0] - px
            dy = spots_xy[:, 1] - py
            dist_sq_min = float((dx * dx + dy * dy).min())

            if dist_sq_min <= tol_sq:
                if proteina == 'Reference':
                    score += 5.0  # bonus alto: las refs deben coincidir
                elif proteina == 'PBS':
                    score += 2.0
                else:
                    score += 1.0

        return score

    # ------------------------------------------------------------------
    def _referencias_por_bandas(
        self, strip_gris: np.ndarray,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        Estrategia PRIMARIA de detección de referencias.

        Los Reference Spots ARY009 están SIEMPRE en:
          - Banda superior (filas 1-2): un blob a la derecha (A) y otro a la izquierda (E)
          - Banda inferior (filas 23-24): un blob alineado con el de columna A (arriba)

        Busca los dos blobs más grandes y más separados horizontalmente
        en la banda superior, luego el blob inferior más alineado en X.
        """
        h, w = strip_gris.shape

        # Rechazar spots demasiado cerca del borde lateral (artefactos del scanner)
        edge_margin = max(5, int(w * 0.03))

        def _spot_no_es_borde(s):
            return edge_margin <= s['cx'] <= w - edge_margin

        # Evaluar TODAS las bandas y elegir la mejor combinación TL/TR/BL.
        # Score = área_promedio * factor_separación * factor_alineación
        # Más alto = mejor (refs grandes, bien separadas horizontalmente, BL bien alineado)
        mejor_score = 0.0
        mejor_combo = None  # (tl_c, tr_c, bp, banda_pct, log_info)

        for band_pct in [0.18, 0.22, 0.28, 0.34]:
            band_h = max(20, int(h * band_pct))
            top_band = strip_gris[:band_h, :]
            bot_band = strip_gris[h - band_h:, :]
            top_spots_raw = self._detectar_spots_para_referencias(top_band, estricto=False)
            bot_spots_raw = self._detectar_spots_para_referencias(bot_band, estricto=False)

            # Filtrar spots de los bordes laterales
            top_spots = [s for s in top_spots_raw if _spot_no_es_borde(s)]
            bot_spots = [s for s in bot_spots_raw if _spot_no_es_borde(s)]

            if len(top_spots) < 2 or not bot_spots:
                continue

            # Ajustar coordenada Y de los spots inferiores
            for s in bot_spots:
                s['cy'] += (h - band_h)

            # Top N candidatos (los más grandes)
            top_srt = sorted(top_spots, key=lambda s: s['area'], reverse=True)
            candidates_top = top_srt[:min(8, len(top_srt))]

            bot_srt = sorted(bot_spots, key=lambda s: s['area'], reverse=True)
            candidates_bot = bot_srt[:min(5, len(bot_srt))]

            # Probar todas las combinaciones (TL, TR) y para cada una el mejor BL
            for ii in range(len(candidates_top)):
                for jj in range(ii + 1, len(candidates_top)):
                    si, sj = candidates_top[ii], candidates_top[jj]
                    sep = abs(si['cx'] - sj['cx'])
                    # Separación mínima: 35% del ancho
                    if sep < w * 0.35:
                        continue

                    if si['cx'] < sj['cx']:
                        tl_c, tr_c = si, sj
                    else:
                        tl_c, tr_c = sj, si

                    # Mejor BL para esta combinación
                    for bp in candidates_bot:
                        d_to_tl = abs(bp['cx'] - tl_c['cx'])
                        d_to_tr = abs(bp['cx'] - tr_c['cx'])
                        align = min(d_to_tl, d_to_tr)
                        lejos = max(d_to_tl, d_to_tr)
                        # Filtro estricto: BL claramente cerca de uno y lejos del otro
                        if align > w * 0.12:
                            continue
                        if align > 0 and lejos / align < 3.0:
                            continue

                        if not self._validar_l_shape(tl_c, tr_c, bp, w, h, estricto=False):
                            continue

                        # Score: favorece (a) referencias grandes, (b) buena separación,
                        # (c) buena alineación vertical, (d) BL razonablemente grande
                        area_avg = (tl_c['area'] + tr_c['area']) / 2.0
                        f_sep = sep / w                  # 0.35-1.0
                        f_align = 1.0 - (align / (w * 0.20))  # 0-1
                        f_bp = min(1.0, bp['area'] / area_avg) if area_avg > 0 else 1.0
                        score = area_avg * f_sep * (0.5 + f_align) * (0.5 + f_bp)

                        if score > mejor_score:
                            mejor_score = score
                            mejor_combo = (
                                tl_c, tr_c, bp, band_pct,
                                f"sep={sep:.0f}px align={align:.0f}px "
                                f"area_top={area_avg:.0f} area_bot={bp['area']:.0f}",
                            )

        if mejor_combo is None:
            logger.info("Bandas: ninguna combinación válida en ninguna banda.")
            return None

        tl_c, tr_c, bp, band_pct, info = mejor_combo
        logger.info(
            f"Referencias por BANDAS (band={band_pct*100:.0f}%, score={mejor_score:.0f}): "
            f"{info}  TL=({tl_c['cx']:.0f},{tl_c['cy']:.0f})  "
            f"TR=({tr_c['cx']:.0f},{tr_c['cy']:.0f})  "
            f"BL=({bp['cx']:.0f},{bp['cy']:.0f})"
        )
        return self._asignar_anchors(tl_c, tr_c, bp)

    # ------------------------------------------------------------------
    @staticmethod
    def _clahe_tile_size(h: int, w: int) -> int:
        """Tile size adaptativo para CLAHE: evita artefactos en strips pequeños."""
        return max(4, min(8, h // 16, w // 4))

    @staticmethod
    def _asignar_spots_uno_a_uno(
        geo_posiciones: Dict[Tuple[str, int], Tuple[int, int]],
        spots_detectados: List[Tuple[float, float, float]],
        max_snap_x: float,
        max_snap_y: float,
        area_min_snap: float,
    ) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """
        Empareja posiciones de la grilla con spots detectados, 1 a 1.

        Cada posición recibe como máximo un spot, y cada spot se entrega a
        como máximo una posición. Esto es lo que impide que dos proteínas
        distintas reclamen el mismo blob: antes el snap se resolvía posición
        por posición, sin memoria de lo ya tomado, y las dos terminaban
        reportando la misma intensidad inventada.

        Estrategia: generar todos los pares (posición, spot) que caen dentro
        de la ventana de búsqueda, ordenarlos por costo y asignarlos de menor
        a mayor, saltando los que ya tienen dueño. El costo es la distancia al
        cuadrado más una penalización para los spots de área pequeña
        (probable ruido), igual que el snap anterior.

        Las posiciones sin spot asignado NO aparecen en el resultado: el
        llamador las deja en su coordenada geométrica.
        """
        if not spots_detectados:
            return {}

        candidatos = []
        for k, (px, py) in geo_posiciones.items():
            for idx, (sx, sy, sa) in enumerate(spots_detectados):
                dx = abs(sx - px)
                dy = abs(sy - py)
                if dx > max_snap_x or dy > max_snap_y:
                    continue
                score = dx * dx + dy * dy
                if sa < area_min_snap:
                    score += ((area_min_snap - sa) / area_min_snap) * (max_snap_x ** 2)
                candidatos.append((score, k, idx))

        # Orden estable: el costo manda, y ante empate decide la coordenada
        # de grilla para que dos corridas sobre la misma imagen coincidan.
        candidatos.sort(key=lambda t: (t[0], t[1], t[2]))

        asignacion: Dict[Tuple[str, int], Tuple[int, int]] = {}
        spots_usados = set()
        for _score, k, idx in candidatos:
            if k in asignacion or idx in spots_usados:
                continue
            sx, sy, _sa = spots_detectados[idx]
            asignacion[k] = (int(round(sx)), int(round(sy)))
            spots_usados.add(idx)
        return asignacion

    @staticmethod
    def _nearest_spot(
        spots_xy: np.ndarray, px: float, py: float,
    ) -> Tuple[int, float]:
        """Índice y dist² del spot más cercano a (px, py) en spots_xy (N×2)."""
        dx = spots_xy[:, 0] - px
        dy = spots_xy[:, 1] - py
        d_sq = dx * dx + dy * dy
        min_idx = int(np.argmin(d_sq))
        return min_idx, float(d_sq[min_idx])

    # ------------------------------------------------------------------
    def _detectar_spots_para_referencias(
        self, strip_gris: np.ndarray, estricto: bool = True,
    ) -> List[Dict]:
        """Detecta candidatos a Reference Spot con filtros configurables."""
        h, w = strip_gris.shape

        # tile size adaptativo: strips pequeños → tiles más grandes para evitar artefactos
        tile_size = self._clahe_tile_size(h, w)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
        eq = clahe.apply(strip_gris)

        # ADAPTATIVO: kernel Black-Hat proporcional al tamaño esperado de spot.
        # Spot diámetro aproximado = 60% del row_spacing = 60% * h / 24
        # Kernel debe ser 2-3x el diámetro del spot
        spot_diam = max(3.0, (h / 24.0) * 0.6)
        tam_k = int(round(spot_diam * 2.5))
        if tam_k % 2 == 0:
            tam_k += 1
        # Límites: entre 5 y 51 píxeles
        tam_k = max(5, min(tam_k, 51))
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_k, tam_k))
        
        max_val = np.iinfo(strip_gris.dtype).max if strip_gris.dtype.kind == 'u' else 255
        fondo_claro = np.mean(strip_gris) > (max_val * 0.35)
        if fondo_claro:
            bhat = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, kernel_bh)
        else:
            bhat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, kernel_bh)

        _, thresh = cv2.threshold(bhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3, iterations=2)
        if estricto:
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3, iterations=1)

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Área mínima ADAPTATIVA: spot ~ (diam)² * π/4, escalado por imagen
        spot_diam_aprox = max(3.0, (h / 24.0) * 0.6)
        spot_area_aprox = spot_diam_aprox * spot_diam_aprox * 0.785
        # Mínimo: 30% del área esperada en estricto, 15% en permisivo
        area_min = max(2, int(spot_area_aprox * (0.30 if estricto else 0.15)))
        circ_min = 0.30 if estricto else 0.15
        spots: List[Dict] = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area < area_min or area > (h * w) / 25:
                continue
            perimetro = cv2.arcLength(c, True)
            if perimetro < 1:
                continue
            circ = 4 * np.pi * area / (perimetro ** 2)
            if circ < circ_min:
                continue
            M = cv2.moments(c)
            if M['m00'] > 0:
                spots.append({
                    'cx': M['m10'] / M['m00'],
                    'cy': M['m01'] / M['m00'],
                    'area': area,
                    'circ': circ,
                })
        return spots

    # ------------------------------------------------------------------
    @staticmethod
    def _validar_l_shape(
        tl_c: Dict, tr_c: Dict, bp: Dict, w: int, h: int,
        estricto: bool = True,
    ) -> bool:
        """
        Verifica que los 3 anchors detectados formen un L-shape consistente
        con la geometría ARY009: TL y BL deben estar en la misma columna A,
        y TR debe estar en la columna E (opuesta).

        Si estricto=True: aspect ratio entre 2.5 y 12.0, áreas similares.
        Si estricto=False: solo verificación geométrica básica.

        CRÍTICO: BL debe estar CLARAMENTE alineado con TL o TR, no en el medio
        (un BL ambiguo causa una grilla diagonal con descuadre acumulativo).
        """
        sep_x = abs(tr_c['cx'] - tl_c['cx'])
        sep_y_left = abs(bp['cy'] - tl_c['cy'])
        sep_y_right = abs(bp['cy'] - tr_c['cy'])
        sep_y = max(sep_y_left, sep_y_right)

        # Separaciones mínimas ADAPTATIVAS al tamaño del strip
        # Para strips chicos (130px alto) sep_y debe ser >40% del alto
        # Para strips grandes (700px alto) >40% es ~280px (igualmente válido)
        min_sep_x = max(10, int(w * 0.30))  # 30% del ancho del strip
        min_sep_y = max(30, int(h * 0.45))  # 45% del alto del strip

        if sep_x < min_sep_x or sep_y < min_sep_y:
            return False

        # ── VERIFICACIÓN CRÍTICA: alineación vertical inequívoca ──
        # BL debe estar SUSTANCIALMENTE más cerca de uno de los anchors top
        # que del otro. Si está en el medio, el grid quedará diagonal.
        d_to_tl = abs(bp['cx'] - tl_c['cx'])
        d_to_tr = abs(bp['cx'] - tr_c['cx'])
        cerca = min(d_to_tl, d_to_tr)
        lejos = max(d_to_tl, d_to_tr)

        # BL debe estar muy cerca de UNO de los anchors superiores
        # Umbral: la "cercana" debe ser <= 12% del ancho del strip
        if cerca > w * 0.12:
            return False

        # Y la "lejana" debe ser claramente más lejana (ratio >= 3x)
        # Si están a distancias similares, BL está ambiguamente en el medio
        if cerca > 0 and lejos / cerca < 3.0:
            return False

        if not estricto:
            return True

        ratio = sep_y / sep_x
        if ratio < 2.5 or ratio > 12.0:
            return False

        areas = [tl_c['area'], tr_c['area'], bp['area']]
        a_min, a_max = min(areas), max(areas)
        if a_min > 0 and a_max / a_min > 6.0:
            return False

        return True

    # ------------------------------------------------------------------
    def _referencias_por_pares(
        self, strip_gris: np.ndarray,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        Estrategia por pares verticales (Reference Spots vienen en pares).
        Genera todas las combinaciones candidatas, las puntúa, y selecciona
        la mejor que pase la validación geométrica del L-shape.
        """
        h, w = strip_gris.shape

        for estricto in (True, False):
            spots = self._detectar_spots_para_referencias(strip_gris, estricto=estricto)
            if len(spots) < 6:
                continue

            # Bounding box dinámico de todos los spots detectados
            min_y = min(s['cy'] for s in spots)
            max_y = max(s['cy'] for s in spots)
            strip_h = max(max_y - min_y, 10.0)
            
            # Ancho dinámico para tolerancias
            min_x = min(s['cx'] for s in spots)
            max_x = max(s['cx'] for s in spots)
            strip_w = max(max_x - min_x, 10.0)

            fila_espacio = strip_h / 23.0
            pairs: List[Dict] = []
            used = set()

            # Criterios MÁS ESTRICTOS para pares verticales relativos al espacio real:
            for i in range(len(spots)):
                if i in used:
                    continue
                s_i = spots[i]
                best_j, best_score = None, float('inf')
                for j in range(len(spots)):
                    if i == j or j in used:
                        continue
                    s_j = spots[j]
                    dx = abs(s_i['cx'] - s_j['cx'])
                    dy = abs(s_i['cy'] - s_j['cy'])
                    if (dx < fila_espacio * 0.45
                            and fila_espacio * 0.55 < dy < fila_espacio * 1.55):
                        score = dx * 3.0 + abs(dy - fila_espacio)
                        if score < best_score:
                            best_score = score
                            best_j = j
                if best_j is not None:
                    s_j = spots[best_j]
                    pairs.append({
                        'cx': (s_i['cx'] + s_j['cx']) / 2.0,
                        'cy': (s_i['cy'] + s_j['cy']) / 2.0,
                        'area': s_i['area'] + s_j['area'],
                    })
                    used.add(i)
                    used.add(best_j)
            logger.info(f"Pares verticales detectados: {len(pairs)}")

            if len(pairs) < 3:
                continue

            cand_estrictos: List[Tuple[float, Dict, Dict, Dict]] = []
            cand_permisivos: List[Tuple[float, Dict, Dict, Dict]] = []

            for zona_top, zona_bot in [(0.30, 0.70), (0.40, 0.60), (0.50, 0.50)]:
                top_pairs = sorted(
                    [p for p in pairs if p['cy'] < min_y + strip_h * zona_top],
                    key=lambda p: p['area'], reverse=True,
                )[:8]
                bot_pairs = sorted(
                    [p for p in pairs if p['cy'] > min_y + strip_h * zona_bot],
                    key=lambda p: p['area'], reverse=True,
                )[:8]
                if len(top_pairs) < 2 or len(bot_pairs) < 1:
                    continue

                for sep_min, align_max in [(0.40, 0.20), (0.30, 0.30), (0.20, 0.40)]:
                    for i in range(len(top_pairs)):
                        for j in range(i + 1, len(top_pairs)):
                            t1, t2 = top_pairs[i], top_pairs[j]
                            sep_x = abs(t1['cx'] - t2['cx'])
                            if sep_x < strip_w * sep_min:
                                continue
                            tl_c, tr_c = (t1, t2) if t1['cx'] < t2['cx'] else (t2, t1)
                            for bp in bot_pairs:
                                d_tl = abs(bp['cx'] - tl_c['cx'])
                                d_tr = abs(bp['cx'] - tr_c['cx'])
                                align = min(d_tl, d_tr)
                                if align > strip_w * align_max:
                                    continue

                                if not self._validar_l_shape(
                                        tl_c, tr_c, bp, int(strip_w), int(strip_h), estricto=False):
                                    continue

                                # Bonus por estar en los extremos del clúster
                                ext_left = max(0.0, tl_c['cx'] - (min_x + strip_w * 0.05))
                                ext_right = max(0.0, (min_x + strip_w * 0.95) - tr_c['cx'])
                                ext_top = max(0.0, min(tl_c['cy'], tr_c['cy']) - (min_y + strip_h * 0.05))
                                ext_bot = max(0.0, (min_y + strip_h * 0.95) - bp['cy'])
                                penalizacion_extremos = (
                                    ext_left + ext_right + ext_top + ext_bot
                                )

                                score = (
                                    (tl_c['area'] + tr_c['area'] + bp['area']) / 3.0
                                    - align * 8.0
                                    + sep_x * 0.6
                                    - penalizacion_extremos * 0.5
                                )

                                if self._validar_l_shape(
                                        tl_c, tr_c, bp, int(strip_w), int(strip_h), estricto=True):
                                    cand_estrictos.append((score, tl_c, tr_c, bp))
                                else:
                                    cand_permisivos.append((score, tl_c, tr_c, bp))

            elegidos = cand_estrictos if cand_estrictos else cand_permisivos
            if elegidos:
                elegidos.sort(key=lambda x: x[0], reverse=True)
                if not cand_estrictos:
                    logger.info("Usando candidato con validación permisiva.")
                _, tl_c, tr_c, bp = elegidos[0]
                return self._asignar_anchors(tl_c, tr_c, bp)

        return None

    # ------------------------------------------------------------------
    def _referencias_legacy(
        self, strip_gris: np.ndarray,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        Estrategia FALLBACK (última línea de defensa):
        Busca el blob más grande en regiones definidas como fracción del
        TAMAÑO REAL del strip (no del bounding box de spots detectados,
        que puede estar desplazado si hay pocos spots visibles).

        Prueba múltiples combinaciones de tamaños de banda top/bottom y
        ancho de la región de búsqueda lateral.
        """
        h, w = strip_gris.shape
        spots = self._detectar_spots_para_referencias(strip_gris, estricto=False)
        if len(spots) < 3:
            return None

        def mayor_en_imagen(x0_frac, x1_frac, y0_frac, y1_frac):
            """Mayor blob en región definida como fracción del strip real (h, w)."""
            cands = [
                b for b in spots
                if w * x0_frac <= b['cx'] <= w * x1_frac
                and h * y0_frac <= b['cy'] <= h * y1_frac
            ]
            return max(cands, key=lambda b: b['area']) if cands else None

        candidatos = []
        # Probar distintos tamaños de banda top/bottom y de zona lateral
        for top_frac in [0.20, 0.25, 0.30, 0.35]:
            for side_frac in [0.45, 0.55, 0.65]:
                tl = mayor_en_imagen(0, side_frac, 0, top_frac)
                tr = mayor_en_imagen(1 - side_frac, 1.0, 0, top_frac)
                bl = mayor_en_imagen(0, side_frac, 1 - top_frac, 1.0)
                br = mayor_en_imagen(1 - side_frac, 1.0, 1 - top_frac, 1.0)

                for bot_c in (bl, br):
                    if tl and tr and bot_c:
                        candidatos.append((tl, tr, bot_c))

        # Validación estricta primero
        for tl, tr, bp in candidatos:
            if self._validar_l_shape(tl, tr, bp, w, h, estricto=True):
                return self._asignar_anchors(tl, tr, bp)

        # Luego permisiva
        for tl, tr, bp in candidatos:
            if self._validar_l_shape(tl, tr, bp, w, h, estricto=False):
                logger.info("Legacy: usando candidato con validación permisiva.")
                return self._asignar_anchors(tl, tr, bp)

        logger.warning("Legacy: no se encontró un L-shape válido.")
        return None

    # ------------------------------------------------------------------
    # Relación col_spacing / row_spacing medida en las membranas ARY009 bien
    # ancladas del set de prueba: 1.29–1.71, estable de 1x a 4x. Es una
    # propiedad del kit (la separación entre columnas es ~1.3 veces la
    # separación entre filas), no del encuadre, así que detecta anclajes
    # imposibles sin depender de la escala, el recorte ni la inclinación.
    RATIO_CELDA_MIN = 0.90
    RATIO_CELDA_MAX = 2.20

    @staticmethod
    def _ratio_celda(
        refs: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        """
        Devuelve (col_spacing, row_spacing, ratio) de unos anclajes.
        ratio vale 0.0 si la geometría es degenerada (algún spacing nulo).
        """
        tl, tr, bl = refs
        col_spacing = abs(tr[0] - tl[0]) / 4.0
        row_spacing = abs(bl[1] - tl[1]) / 22.0
        if col_spacing <= 0 or row_spacing <= 0:
            return col_spacing, row_spacing, 0.0
        return col_spacing, row_spacing, col_spacing / row_spacing

    @classmethod
    def _geometria_plausible(
        cls,
        refs: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        n_strip: int,
    ) -> bool:
        """
        Rechaza anclajes que no pueden corresponder a una membrana ARY009.

        Cuando la detección de Reference Spots se equivoca, el grid resultante
        es geométricamente imposible pero el motor lo usa igual y entrega
        números con apariencia normal. Dos modos de falla observados:

          - Anclas superiores demasiado juntas → las 5 columnas colapsan sobre
            la misma X y varias proteínas miden el mismo spot.
          - Anclas superiores demasiado separadas → el grid se abre más que la
            membrana y las columnas de los bordes caen fuera de los spots.

        Ambos se detectan por la forma de la celda (col_spacing / row_spacing),
        que en una membrana real vale ~1.3 sin importar el tamaño de la foto.
        """
        col_spacing, row_spacing, ratio = cls._ratio_celda(refs)

        if ratio == 0.0:
            logger.warning(
                f"Strip {n_strip}: anclajes degenerados "
                f"(col_spacing={col_spacing:.1f}px, row_spacing={row_spacing:.1f}px). "
                f"Se descarta la tirilla."
            )
            return False

        if not (cls.RATIO_CELDA_MIN <= ratio <= cls.RATIO_CELDA_MAX):
            forma = "columnas colapsadas" if ratio < cls.RATIO_CELDA_MIN else "columnas sobre-extendidas"
            logger.warning(
                f"Strip {n_strip}: geometría inválida — {forma}. "
                f"La celda mide {col_spacing:.1f}x{row_spacing:.1f}px (relación {ratio:.2f}, "
                f"se esperaba entre {cls.RATIO_CELDA_MIN} y {cls.RATIO_CELDA_MAX}). "
                f"Los Reference Spots se detectaron mal: las intensidades de esta "
                f"tirilla no serían confiables. Recalibre manualmente."
            )
            return False
        return True

    # ------------------------------------------------------------------
    def _asignar_anchors(
        self, top_left: Dict, top_right: Dict, bot_pair: Dict,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Asigna los 3 anchors detectados a (tl, tr, bl) de _pos_pixel
        según la orientación detectada (col A izquierda o derecha).
        """
        aligned_with_left = (
            abs(bot_pair['cx'] - top_left['cx'])
            < abs(bot_pair['cx'] - top_right['cx'])
        )
        if aligned_with_left:
            tl = (top_left['cx'], top_left['cy'])
            tr = (top_right['cx'], top_right['cy'])
            bl = (bot_pair['cx'], bot_pair['cy'])
            logger.info("Orientación: col A en la IZQUIERDA del strip.")
        else:
            tl = (top_right['cx'], top_right['cy'])
            tr = (top_left['cx'], top_left['cy'])
            bl = (bot_pair['cx'], bot_pair['cy'])
            logger.info("Orientación: col A en la DERECHA del strip.")

        tl, tr, bl = self._centrar_en_pares(tl, tr, bl)

        logger.info(
            f"Referencias: TL=({tl[0]:.0f},{tl[1]:.0f})  "
            f"TR=({tr[0]:.0f},{tr[1]:.0f})  "
            f"BL=({bl[0]:.0f},{bl[1]:.0f})"
        )
        return (tl, tr, bl)

    # ------------------------------------------------------------------
    def _centrar_en_pares(self, tl, tr, bl):
        """Lleva cada ancla al centro de su PAR de duplicados.

        Las 3 anclas del ARY009 son pares verticales (A1/A2, E1/E2, A23/A24) y
        _pos_pixel las espera en el CENTRO del par: _FILA_REF_TOP = 1.5 y
        _FILA_REF_BOT = 23.5. Cuando los dos spots del par se fusionan en un
        blob, su centroide ya es ese centro. Pero si el par sale como dos blobs
        separados —pasa cuando el valle entre ellos es profundo, o cuando uno
        de los dos está tenue— la estrategia elige UNO y el ancla queda media
        fila corrida.

        Esa media fila no la delata la relación de celda: la grilla conserva su
        forma y sigue dando ~1.3. Lo que sí se rompe es que el compañero del
        par se desploma (A23 medido en 13.5 contra A24 en 54.2), el PBS D23/D24
        se monta sobre la cola del spot de arriba y se resta como fondo a las
        35 proteínas, y el paso de fila se descuadra hasta colapsar dos filas
        en el mismo píxel. Era el estado de Sw620 colon-mango M2.

        El arreglo se apoya en que el detector general de spots SÍ encuentra el
        par fusionado: en Sw620 M2 hay un spot de área 144 en y=301.7 (el
        centro real del par) a 5.5 px del ancla que la estrategia había puesto
        en y=296.2 (el centro de un solo spot). Las estrategias de referencias
        usan su propio detector y pueden quedarse con un sub-blob.

        Así que entre los spots cercanos al ancla se elige el de MAYOR ÁREA: un
        par fusionado mide más que uno de sus dos spots sueltos. Si el ancla ya
        estaba sobre el par, el mayor cercano es él mismo y no se mueve nada.
        """
        spots = getattr(self, '_spots_para_anclas', None)
        if not spots:
            return tl, tr, bl

        # Paso de fila a partir de las propias anclas: van de la fila 1.5 a la
        # 23.5, o sea 22 pasos.
        rs = abs(bl[1] - tl[1]) / 22.0
        if rs <= 0:
            return tl, tr, bl
        # Radio de búsqueda por debajo de 1 paso de fila, para no poder saltar
        # nunca al par de otra fila, y por debajo de media celda en horizontal,
        # para no invadir la columna vecina (col_spacing es ~1.3 rs).
        tol_y = rs * 0.75
        tol_x = rs * 0.55

        def centrar(anc, nombre):
            ax, ay = anc
            mejor = None
            for sx, sy, area in spots:
                if abs(sx - ax) > tol_x or abs(sy - ay) > tol_y:
                    continue
                if mejor is None or area > mejor[0]:
                    mejor = (area, sx, sy)
            if mejor is None:
                return anc
            area, sx, sy = mejor
            if abs(sx - ax) < 0.5 and abs(sy - ay) < 0.5:
                return anc
            logger.info(
                f"  Ancla {nombre} enganchada al blob del par: "
                f"({ax:.0f},{ay:.0f}) -> ({sx:.0f},{sy:.0f}) "
                f"(area={area:.0f}, paso de fila {rs:.1f}px)"
            )
            return (sx, sy)

        return centrar(tl, 'TL'), centrar(tr, 'TR'), centrar(bl, 'BL')

    # ------------------------------------------------------------------
    def _pos_pixel(
        self,
        col_letra: str,
        fila: int,
        tl: Tuple[float, float],
        tr: Tuple[float, float],
        bl: Tuple[float, float],
        pbs_bot: Optional[Tuple[float, float]] = None,
        img_inv: Optional[np.ndarray] = None,
        rx: int = 0,
        ry: int = 0,
    ) -> Tuple[int, int]:
        """
        Interpola la posición en píxeles de un spot.
        Aplica corrección trapezoidal si pbs_bot es proporcionado.
        """
        t_col = self._T_COLS[col_letra]
        t_row = (fila - self._FILA_REF_TOP) / (self._FILA_REF_BOT - self._FILA_REF_TOP)

        # ── Y por fila ──
        # En MODO RÍGIDO: bloqueamos Y para que sea IDÉNTICO entre todas las
        # columnas de una misma fila (restricción geométrica del kit ARY009:
        # A23, B23, C23, D23 y E23 están todos a la MISMA altura).
        # En modo adaptativo: permitimos pequeña corrección por inclinación TR↔TL.
        if getattr(self, 'modo_rigido', False):
            py = tl[1] + t_row * (bl[1] - tl[1])
        else:
            py = tl[1] + t_row * (bl[1] - tl[1]) + t_col * (tr[1] - tl[1])

        x_A = tl[0] + t_row * (bl[0] - tl[0])

        if pbs_bot is not None:
            # pbs_bot es D23/D24 (t_col = _T_COLS['D'], t_row = 1.0)
            t_col_D = self._T_COLS['D']
            x_D_top = tl[0] + t_col_D * (tr[0] - tl[0])
            x_D_bot = pbs_bot[0]
            x_D = x_D_top + t_row * (x_D_bot - x_D_top)

            w_total = (x_D - x_A) / t_col_D
            px_float = x_A + t_col * w_total
        else:
            px_float = x_A + t_col * (tr[0] - tl[0])

        px, py = int(round(px_float)), int(round(py))

        # =========================================================
        # Safe Snapping iterativo (Mean-Shift Limitado)
        # 3 iteraciones con ROI progresivamente más pequeño:
        #   pasada 1: ROI amplio (atrae spots lejanos a la grilla)
        #   pasada 2: ROI medio (refina sobre el blob detectado)
        #   pasada 3: ROI pequeño (centra finamente en el pico)
        # =========================================================
        if img_inv is not None and rx > 0 and ry > 0:
            h, w = img_inv.shape
            px_orig, py_orig = px, py

            # Límites físicos: aumentamos el margen al 16% del ancho total
            # para capturar distorsión de lente en las columnas C y D (ej. D1, D2, C1..C4)
            max_shift_x = max(2, int(abs(tr[0] - tl[0]) * 0.16))
            row_spacing = abs(bl[1] - tl[1]) / 23.0 if abs(bl[1] - tl[1]) > 0 else ry * 2
            max_shift_y = max(2, int(row_spacing * 0.65))

            # Factores de tamaño de ROI por pasada (atrae → refina → centra)
            roi_factors = [1.20, 0.85, 0.55]
            # Umbrales decrecientes para capturar también manchas tenues
            thresh_factors = [1.18, 1.12, 1.08]

            prev_px, prev_py = px, py
            for it, (rf, tf) in enumerate(zip(roi_factors, thresh_factors)):
                rxx = max(2, int(rx * rf))
                ryy = max(2, int(ry * rf))
                x0, x1 = max(0, px - rxx), min(w, px + rxx)
                y0, y1 = max(0, py - ryy), min(h, py + ryy)
                roi = img_inv[y0:y1, x0:x1]

                if roi.size == 0:
                    break

                max_val = float(roi.max())
                mean_val = float(roi.mean())
                # Umbral mínimo: requerir contraste real con el fondo
                if max_val < mean_val * tf or max_val < 12:
                    continue

                # Centroide ponderado por la intensidad sobre la media
                roi_weights = np.clip(roi.astype(np.float32) - mean_val, 0, None)
                sum_weights = float(np.sum(roi_weights))
                if sum_weights <= 0:
                    continue
                y_idx, x_idx = np.indices(roi_weights.shape)
                cx_roi = float(np.sum(x_idx * roi_weights) / sum_weights)
                cy_roi = float(np.sum(y_idx * roi_weights) / sum_weights)

                px_new = x0 + int(round(cx_roi))
                py_new = y0 + int(round(cy_roi))

                # Clamping estricto desde la posición geométrica original
                px = int(np.clip(px_new,
                                  px_orig - max_shift_x,
                                  px_orig + max_shift_x))
                py = int(np.clip(py_new,
                                  py_orig - max_shift_y,
                                  py_orig + max_shift_y))

                # Convergencia: si el movimiento es <= 1 px, terminar
                if abs(px - prev_px) <= 1 and abs(py - prev_py) <= 1 and it >= 1:
                    break
                prev_px, prev_py = px, py

        return px, py

    # ------------------------------------------------------------------
    def _detectar_todos_los_spots(
        self, strip_gris: np.ndarray,
    ) -> List[Tuple[float, float, float]]:
        """
        Detecta TODAS las manchas oscuras del strip combinando MÚLTIPLES
        escalas de Black-Hat y un umbral adaptativo bajo. Esto captura
        tanto spots grandes (Reference) como tenues/pequeños (proteínas
        con baja expresión) que un único kernel se perdería.

        Returns:
            Lista de (cx, cy, area) para cada spot detectado.
        """
        # Validar que el strip no sea trivialmente vacío
        if strip_gris is None or strip_gris.size == 0:
            return []
        if strip_gris.shape[0] < 10 or strip_gris.shape[1] < 10:
            return []

        # Normalizar a 8-bit para que los umbrales sean estables
        if strip_gris.dtype != np.uint8:
            gris_8 = cv2.normalize(strip_gris, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
        else:
            gris_8 = strip_gris.copy()

        h_strip, w_strip = gris_8.shape

        # ADAPTATIVO: los kernels deben escalar con el tamaño del strip.
        # Estimación de tamaño del spot: row_spacing aprox = h/24 (24 filas ARY009)
        # Spot diámetro ≈ 60% del row spacing
        row_spacing_aprox = h_strip / 24.0
        spot_diam_aprox = max(3.0, row_spacing_aprox * 0.6)

        # Kernel Black-Hat debe ser 1.5x-3x el diámetro del spot
        # Generamos 4 kernels en este rango
        kernels_sizes = []
        for factor in (1.4, 2.0, 2.8, 3.8):
            k = int(round(spot_diam_aprox * factor))
            # Forzar impar y mínimo 5
            if k % 2 == 0:
                k += 1
            k = max(5, k)
            if k not in kernels_sizes:
                kernels_sizes.append(k)

        # Suavizado: kernel proporcional pero al menos 3x3
        blur_k = 3 if row_spacing_aprox < 15 else 5
        gris_8 = cv2.GaussianBlur(gris_8, (blur_k, blur_k), 0)

        # Multi-escala adaptativa
        mascara_combinada = None
        for k_size in kernels_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            bhat = cv2.morphologyEx(gris_8, cv2.MORPH_BLACKHAT, kernel)
            if bhat.max() < 4:
                continue
            umbral = max(3.0, float(bhat.max()) * 0.09)
            _, m = cv2.threshold(bhat, umbral, 255, cv2.THRESH_BINARY)
            if mascara_combinada is None:
                mascara_combinada = m
            else:
                mascara_combinada = cv2.bitwise_or(mascara_combinada, m)

        # Fallback para imágenes de bajo contraste: si no se detectó nada o muy poco,
        # intentar de nuevo con CLAHE que amplifica el contraste local.
        n_detectados = 0 if mascara_combinada is None else int(cv2.countNonZero(mascara_combinada))
        if n_detectados < 5:
            tile_size = self._clahe_tile_size(h_strip, w_strip)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile_size, tile_size))
            gris_clahe = clahe.apply(gris_8)
            for k_size in kernels_sizes:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                bhat = cv2.morphologyEx(gris_clahe, cv2.MORPH_BLACKHAT, kernel)
                if bhat.max() < 2:
                    continue
                umbral = max(1.5, float(bhat.max()) * 0.06)
                _, m = cv2.threshold(bhat, umbral, 255, cv2.THRESH_BINARY)
                mascara_combinada = m if mascara_combinada is None else cv2.bitwise_or(mascara_combinada, m)
            if mascara_combinada is not None and cv2.countNonZero(mascara_combinada) > n_detectados:
                logger.info("Spots: fallback CLAHE activado (contraste bajo en Black-Hat estándar)")

        if mascara_combinada is None:
            return []

        # Limpieza para eliminar ruido sub-píxel
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mascara_combinada = cv2.morphologyEx(
            mascara_combinada, cv2.MORPH_OPEN, k3, iterations=1)
        mascara_combinada = cv2.morphologyEx(
            mascara_combinada, cv2.MORPH_CLOSE, k3, iterations=1)

        contornos, _ = cv2.findContours(
            mascara_combinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centros: List[Tuple[float, float, float]] = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area < 3:
                continue
            M = cv2.moments(c)
            if M['m00'] < 1:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centros.append((cx, cy, area))
        return centros

    # ------------------------------------------------------------------
    def _icp_refinar(
        self,
        pos_dict: Dict[Tuple[str, int], Tuple[int, int]],
        spots: List[Tuple[float, float, float]],
        row_spacing: float,
        col_spacing: float,
        max_iter: int = 3,
    ) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """
        Refinamiento ICP (Iterative Closest Point):
        Encuentra la transformación de similaridad (rotación + traslación +
        escala uniforme) que mejor alinea la grilla a los spots detectados.

        Es robusto a:
          - Membrana ligeramente rotada
          - Escala global del grid distinta a la esperada
          - Traslación residual

        Aplica safeguards para que la transformación no distorsione la grilla:
          - Escala entre 0.85 y 1.15
          - Rotación máxima ±15°
          - Requiere al menos 8 spots para la vía afín; con 3-7 spots
            degrada a traslación pura por mediana
        """
        if len(spots) < 3:
            return pos_dict

        spots_xy = np.array([[s[0], s[1]] for s in spots], dtype=np.float32)
        max_dist = max(row_spacing, col_spacing) * 0.50
        max_dist_sq = max_dist * max_dist

        pos = dict(pos_dict)

        # Con menos de 8 spots no hay suficiente información para una transformación
        # afín robusta. Hacemos solo traslación: la mediana del desplazamiento de
        # cada posición grilla al spot real más cercano.
        if len(spots) < 8:
            shifts_x, shifts_y = [], []
            for (col, fila), (px, py) in pos.items():
                min_idx, dist_sq = self._nearest_spot(spots_xy, px, py)
                if dist_sq < max_dist_sq:
                    shifts_x.append(float(spots_xy[min_idx, 0] - px))
                    shifts_y.append(float(spots_xy[min_idx, 1] - py))
            if len(shifts_x) >= 3:
                tx = float(np.median(shifts_x))
                ty = float(np.median(shifts_y))
                # Limitar a 30% del espaciado para evitar saltos bruscos
                max_t = max(row_spacing, col_spacing) * 0.30
                tx = float(np.clip(tx, -max_t, max_t))
                ty = float(np.clip(ty, -max_t, max_t))
                if abs(tx) > 0.5 or abs(ty) > 0.5:
                    logger.info(
                        f"ICP fallback traslación ({tx:+.1f}, {ty:+.1f}) px "
                        f"— {len(shifts_x)} matches con {len(spots)} spots"
                    )
                    for k in list(pos.keys()):
                        pos[k] = (int(round(pos[k][0] + tx)),
                                  int(round(pos[k][1] + ty)))
            return pos

        for _ in range(max_iter):
            src_list, dst_list = [], []
            for (col, fila), (px, py) in pos.items():
                min_idx, dist_sq = self._nearest_spot(spots_xy, px, py)
                if dist_sq < max_dist_sq:
                    src_list.append([float(px), float(py)])
                    dst_list.append([float(spots_xy[min_idx, 0]),
                                      float(spots_xy[min_idx, 1])])

            if len(src_list) < 8:
                break

            src = np.array(src_list, dtype=np.float32).reshape(-1, 1, 2)
            dst = np.array(dst_list, dtype=np.float32).reshape(-1, 1, 2)

            try:
                M, _inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=500, confidence=0.95,
                )
            except Exception:
                break
            if M is None:
                break

            # Validar transformación: rechazar distorsiones extremas
            scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
            if not (0.85 < scale < 1.15):
                break
            rotation = float(np.arctan2(M[1, 0], M[0, 0]))
            if abs(rotation) > np.radians(15):
                break

            new_pos = {}
            max_move = 0.0
            for k, (px, py) in pos.items():
                nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
                move = abs(nx - px) + abs(ny - py)
                if move > max_move:
                    max_move = move
                new_pos[k] = (int(round(nx)), int(round(ny)))

            pos = new_pos
            if max_move < 1.5:
                break

        return pos

    # ------------------------------------------------------------------
    def _calcular_calidad(
        self,
        inv_img: np.ndarray,
        px: int, py: int, r: int,
    ) -> float:
        """
        Score de calidad [0-1] de un cuadro de muestreo.
        Mide el contraste local: spot vs fondo cercano.
          - 1.0: cuadro centrado sobre una mancha clara
          - 0.5: cuadro sobre una mancha tenue
          - 0.0: cuadro sobre fondo vacío
        """
        h, w = inv_img.shape
        x0, x1 = max(0, px - r), min(w, px + r)
        y0, y1 = max(0, py - r), min(h, py + r)
        roi = inv_img[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0
        spot_signal = float(roi.mean())

        # Fondo: franja horizontal (mismo Y que el spot, píxeles a izquierda y derecha).
        # Evita incluir spots de filas adyacentes del mismo columna — en ARY009 cada
        # proteína ocupa 2 filas consecutivas separadas ~row_spacing px, y el bg_r
        # vertical cruzaría ese spot vecino, inflando bg_signal y bajando calidad.
        bg_r = r * 3
        bg_left  = inv_img[y0:y1, max(0, px - bg_r):max(0, px - r)]
        bg_right = inv_img[y0:y1, min(w, px + r):min(w, px + bg_r)]
        bg_parts = [a for a in (bg_left, bg_right) if a.size > 0]
        if not bg_parts:
            return 0.5
        bg_signal = float(np.percentile(np.concatenate([p.ravel() for p in bg_parts]), 25))

        contrast = spot_signal - bg_signal
        # Normalizar: contraste >= 25 → calidad 1.0; <= 3 → calidad 0
        quality = min(1.0, max(0.0, (contrast - 3.0) / 22.0))
        return quality

    # ------------------------------------------------------------------
    @staticmethod
    def _aplicar_consistencia_filas(
        posiciones: Dict[Tuple[str, int], Tuple[int, int]],
        tolerancia_y: int,
    ) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """
        Hace que todas las posiciones de la misma fila tengan Y similar.
        Si una posición tiene Y muy diferente del mediano de su fila,
        se considera outlier y se reemplaza por el Y mediano (manteniendo X).
        Esto rescata casos donde un spot real estaba fuera del rango del
        snap pero sus vecinos de fila sí se anclaron correctamente.
        """
        por_fila: Dict[int, List[Tuple[str, int, int]]] = {}
        for (col, fila), (px, py) in posiciones.items():
            por_fila.setdefault(fila, []).append((col, px, py))

        refinado = dict(posiciones)
        for fila, items in por_fila.items():
            if len(items) < 3:
                continue
            ys = [p[2] for p in items]
            mediano_y = int(round(float(np.median(ys))))
            for col, px, py in items:
                if abs(py - mediano_y) > tolerancia_y:
                    refinado[(col, fila)] = (px, mediano_y)
        return refinado

    # ------------------------------------------------------------------
    @staticmethod
    def _aplicar_consistencia_columnas(
        posiciones: Dict[Tuple[str, int], Tuple[int, int]],
        tolerancia_x: int,
    ) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """
        Análogo a consistencia por filas pero para columnas: posiciones
        de la misma columna deben tener X similar.
        """
        por_col: Dict[str, List[Tuple[int, int, int]]] = {}
        for (col, fila), (px, py) in posiciones.items():
            por_col.setdefault(col, []).append((fila, px, py))

        refinado = dict(posiciones)
        for col, items in por_col.items():
            if len(items) < 3:
                continue
            xs = [p[1] for p in items]
            mediano_x = int(round(float(np.median(xs))))
            for fila, px, py in items:
                if abs(px - mediano_x) > tolerancia_x:
                    refinado[(col, fila)] = (mediano_x, py)
        return refinado

    # ------------------------------------------------------------------
    def _calcular_posiciones_finales(
        self,
        strip_gris: np.ndarray,
        tl: Tuple,
        tr: Tuple,
        bl: Tuple,
        pbs_bot: Optional[Tuple] = None,
    ) -> Tuple[Dict[Tuple[str, int], Tuple[int, int]], np.ndarray, int]:
        """
        Calcula las posiciones finales de muestreo para todos los puntos del grid.

        Pipeline de 4 pasadas:
          1. Posición geométrica + mean-shift + snap a spots globales
          2. Consistencia de filas y columnas (medianas)
          2.5. ICP — alineación afín a los spots reales
          3. Re-snap final desde posición refinada

        Returns:
            (posiciones, img_inv, radio):
            - posiciones: Dict {(col, fila): (px, py)}
            - img_inv: imagen invertida usada para muestreo
            - radio: radio de muestreo dinámico
        """
        h, w = strip_gris.shape
        max_val = np.iinfo(strip_gris.dtype).max if strip_gris.dtype.kind == 'u' else 255
        fondo_claro = np.mean(strip_gris) > (max_val * 0.35)
        inv = cv2.bitwise_not(strip_gris) if fondo_claro else strip_gris
        r = self.get_radio_dinamico(tl, bl)

        # ── MODO RÍGIDO: solo interpolación geométrica, sin snap individual ──
        # Cada posición se calcula EXACTAMENTE desde las 3 referencias usando
        # _T_COLS y _FILA_REF_TOP/BOT. PERO aplicamos UN solo shift global
        # (la mediana del desplazamiento de todos los matches con spots reales)
        # para corregir un offset sistemático que pueda venir de las referencias.
        # Esto preserva la rigidez del grid pero corrige la posición global.
        if self.modo_rigido:
            # ── Detectar spots para usar como 4to anchor (PBS) y shift global ──
            spots_detectados = self._detectar_todos_los_spots(strip_gris)

            # ── AUTO-DETECTAR PBS (D23/D24) si no fue provisto ──
            # Si encontramos un spot oscuro cerca de la posición esperada de PBS,
            # lo usamos como 4to anchor para CORRECCIÓN TRAPEZOIDAL. Esto corrige
            # el error acumulativo hacia el extremo opuesto de BL.
            pbs_bot_auto = pbs_bot
            if pbs_bot_auto is None and spots_detectados:
                # Posición esperada de PBS D23-24 con grid puramente geométrico:
                #   x = TL.x + 0.64 * (TR.x - TL.x)  (columna D, t_col=0.64)
                #   y ≈ BL.y (misma fila 23.5 que A_bot)
                t_col_D = self._T_COLS['D']
                pbs_x_esp = tl[0] + t_col_D * (tr[0] - tl[0])
                pbs_y_esp = bl[1]
                # Buscar spot oscuro cerca de esa posición (tolerancia 35% col_spacing)
                col_sp = abs(tr[0] - tl[0]) / 4.0
                row_sp = abs(bl[1] - tl[1]) / 22.0
                tol = max(col_sp, row_sp) * 0.35
                tol_sq = tol * tol
                mejor_pbs = None
                mejor_dist_sq = float('inf')
                for sx, sy, sa in spots_detectados:
                    dist_sq = (sx - pbs_x_esp) ** 2 + (sy - pbs_y_esp) ** 2
                    if dist_sq <= tol_sq and dist_sq < mejor_dist_sq:
                        # Excluir si ese spot es el BL (mismo spot)
                        if abs(sx - bl[0]) > 5 or abs(sy - bl[1]) > 5:
                            mejor_dist_sq = dist_sq
                            mejor_pbs = (float(sx), float(sy))
                if mejor_pbs is not None:
                    pbs_bot_auto = mejor_pbs
                    logger.info(
                        f"Modo RÍGIDO: PBS auto-detectado en {mejor_pbs} "
                        f"(esperado: ({pbs_x_esp:.0f},{pbs_y_esp:.0f})) "
                        f"→ corrección trapezoidal activada"
                    )

            posiciones: Dict[Tuple[str, int], Tuple[int, int]] = {}
            for (col, fila), _proteina in MAPA_PROTEINAS.items():
                px, py = self._pos_pixel(col, fila, tl, tr, bl, pbs_bot_auto,
                                          img_inv=None, rx=0, ry=0)
                posiciones[(col, fila)] = (px, py)

            # spots_detectados ya está computado arriba
            if spots_detectados:
                row_spacing = abs(bl[1] - tl[1]) / 22.0 if abs(bl[1] - tl[1]) > 0 else r * 2
                col_spacing = abs(tr[0] - tl[0]) / 4.0 if abs(tr[0] - tl[0]) > 0 else r * 2
                # Tolerancia para buscar el match: 50% del menor spacing
                tol = max(row_spacing, col_spacing) * 0.50
                spots_xy = np.array([[s[0], s[1]] for s in spots_detectados], dtype=np.float32)

                dxs, dys = [], []
                for k, (px, py) in posiciones.items():
                    diffs = spots_xy - np.array([px, py], dtype=np.float32)
                    dist_sq = (diffs ** 2).sum(axis=1)
                    idx_min = int(np.argmin(dist_sq))
                    if dist_sq[idx_min] <= tol ** 2:
                        dxs.append(float(diffs[idx_min, 0]))
                        dys.append(float(diffs[idx_min, 1]))

                # Aplicar shift global solo si suficientes matches y shift razonable
                if len(dxs) >= 10:
                    shift_x = int(round(float(np.median(dxs))))
                    shift_y = int(round(float(np.median(dys))))
                    # Limitar shift a 30% del row/col spacing (no movimiento bruto)
                    max_shift = int(max(row_spacing, col_spacing) * 0.30)
                    shift_x = max(-max_shift, min(max_shift, shift_x))
                    shift_y = max(-max_shift, min(max_shift, shift_y))
                    if shift_x != 0 or shift_y != 0:
                        logger.info(
                            f"Modo RÍGIDO: shift global ({shift_x:+d}, {shift_y:+d}) px "
                            f"basado en {len(dxs)} matches"
                        )
                        for k in list(posiciones.keys()):
                            posiciones[k] = (posiciones[k][0] + shift_x,
                                              posiciones[k][1] + shift_y)

                # ── SNAP SUAVE (±3 px) ──
                # Después del shift global, permitir un micro-ajuste por posición
                # para corregir distorsiones LOCALES (ej. la parte inferior del
                # strip que tiene compresión horizontal). Preserva la rigidez
                # global del grid pero permite ±3 px por posición.
                snap_radius = 3
                snap_radius_sq = snap_radius * snap_radius
                n_snapped = 0
                for k, (px, py) in list(posiciones.items()):
                    diffs = spots_xy - np.array([px, py], dtype=np.float32)
                    dist_sq = (diffs ** 2).sum(axis=1)
                    idx_min = int(np.argmin(dist_sq))
                    if dist_sq[idx_min] <= snap_radius_sq:
                        # Hay un spot dentro de ±3px → ajustar
                        new_px = int(round(float(spots_xy[idx_min, 0])))
                        new_py = int(round(float(spots_xy[idx_min, 1])))
                        posiciones[k] = (new_px, new_py)
                        n_snapped += 1
                if n_snapped > 0:
                    logger.info(f"Modo RÍGIDO: {n_snapped} posiciones snap-eadas ±{snap_radius}px")

            logger.info(f"Modo RÍGIDO: {len(posiciones)} posiciones geométricas + shift global.")
            return posiciones, inv, r

        # Radios de búsqueda para el Safe Snapping (mean-shift local)
        rx_safe = max(1, int(abs(tr[0] - tl[0]) * 0.08))
        ry_safe = max(1, r)

        # Detección global de spots (multi-escala Black-Hat)
        spots_detectados = self._detectar_todos_los_spots(strip_gris)
        row_spacing = abs(bl[1] - tl[1]) / 22.0 if abs(bl[1] - tl[1]) > 0 else r * 2
        col_spacing = abs(tr[0] - tl[0]) / 4.0 if abs(tr[0] - tl[0]) > 0 else r * 2
        # Umbral de snap: 48% del espaciado entre columnas (A-B) en X,
        # y 45% del espaciado entre filas en Y. Valores amplios para capturar
        # spots que quedaron con offset por _T_COLS o por deformación de membrana.
        max_snap_x = col_spacing * 0.48
        max_snap_y = row_spacing * 0.45

        # Umbral mínimo de área para considerar un spot real (filtra píxeles de ruido)
        area_min_snap = max(6.0, row_spacing * 0.08)

        def _snap_a_spot_cercano(px: int, py: int) -> Tuple[int, int]:
            """
            Snap a la mancha detectada más cercana dentro de los límites.
            Penaliza spots muy pequeños (probable ruido) para preferir
            blobs reales con área significativa.
            """
            if not spots_detectados:
                return px, py
            mejor_spot = None
            mejor_score = float('inf')
            for sx, sy, sa in spots_detectados:
                dx = abs(sx - px)
                dy = abs(sy - py)
                if dx <= max_snap_x and dy <= max_snap_y:
                    dist_sq = dx * dx + dy * dy
                    # Penalización por área pequeña: spots de área < area_min_snap
                    # tienen un costo extra proporcional al déficit de área.
                    if sa < area_min_snap:
                        penalizacion = ((area_min_snap - sa) / area_min_snap) * (max_snap_x ** 2)
                    else:
                        penalizacion = 0.0
                    score = dist_sq + penalizacion
                    if score < mejor_score:
                        mejor_score = score
                        mejor_spot = (sx, sy)
            if mejor_spot is not None:
                return int(round(mejor_spot[0])), int(round(mejor_spot[1]))
            return px, py

        # ── PASADA 1: posición geométrica + snap global ──
        # Guardamos geo_posiciones antes del snap para poder revertir duplicados.
        posiciones: Dict[Tuple[str, int], Tuple[int, int]] = {}
        geo_posiciones: Dict[Tuple[str, int], Tuple[int, int]] = {}
        for (col, fila), proteina in MAPA_PROTEINAS.items():
            gx, gy = self._pos_pixel(col, fila, tl, tr, bl, pbs_bot,
                                     img_inv=inv, rx=rx_safe, ry=ry_safe)
            geo_posiciones[(col, fila)] = (gx, gy)

        # Asignación global 1-a-1 en vez de snap posición por posición: dos
        # proteínas ya no pueden quedarse con el mismo spot. Las posiciones sin
        # spot asignado conservan su coordenada geométrica.
        asignacion = self._asignar_spots_uno_a_uno(
            geo_posiciones, spots_detectados,
            max_snap_x, max_snap_y, area_min_snap,
        )
        for k, geo in geo_posiciones.items():
            posiciones[k] = asignacion.get(k, geo)
        logger.info(
            f"Asignación 1-a-1: {len(asignacion)}/{len(geo_posiciones)} posiciones "
            f"con spot propio ({len(spots_detectados)} spots detectados)"
        )

        # ── PASADA 2: consistencia por filas (Y) y columnas (X) ──
        # Tolerancias algo amplias para no forzar spots que se anclaron
        # correctamente a posiciones ligeramente distintas de sus vecinos.
        # La variación natural (rotación pequeña, distorsión) puede ser de hasta ~35-40%.
        tol_y = max(4, int(row_spacing * 0.40))
        tol_x = max(4, int(col_spacing * 0.32))
        posiciones = self._aplicar_consistencia_filas(posiciones, tol_y)
        posiciones = self._aplicar_consistencia_columnas(posiciones, tol_x)

        # Guardar posiciones de Reference y PBS ANTES del ICP.
        # Estas posiciones ya están bien definidas por los anchors de referencia,
        # así que el ICP (transformación global) puede empeorarlas.
        coords_ancladas = set()
        pos_ancladas = {}
        for (col, fila), proteina in MAPA_PROTEINAS.items():
            if proteina in ('Reference', 'PBS'):
                coords_ancladas.add((col, fila))
                if (col, fila) in posiciones:
                    pos_ancladas[(col, fila)] = posiciones[(col, fila)]

        # ── PASADA 2.5: ICP — alineación afín a los spots reales ──
        posiciones = self._icp_refinar(
            posiciones, spots_detectados, row_spacing, col_spacing,
        )

        # ── PASADA 3: re-asignación final 1-a-1 desde la posición refinada ──
        # Tiene que ser 1-a-1 igual que la PASADA 1: si aquí se volviera a
        # snapear posición por posición, dos proteínas podrían reclamar de
        # nuevo el mismo spot y se perdería lo ganado antes del ICP.
        asignacion_final = self._asignar_spots_uno_a_uno(
            posiciones, spots_detectados,
            max_snap_x, max_snap_y, area_min_snap,
        )
        snapeadas = set()  # coords (col, fila) que quedaron sobre un spot real
        for k in list(posiciones.keys()):
            nueva = asignacion_final.get(k)
            if nueva is not None:
                posiciones[k] = nueva
                snapeadas.add(k)

        # ── DEDUPLICACIÓN post-snap: filas consecutivas del mismo columna ──
        # Si dos posiciones contiguas (e.g. D9 y D10, duplicados del mismo proteína)
        # snapearon al mismo blob, la y_mediano_fila de PASADA 4 usará ese Y único
        # para DOS filas distintas → todas las posiciones de ambas filas colapsan.
        # Fix: revertir el que se alejó más de la posición geométrica.
        for col in sorted(set(c for c, _ in MAPA_PROTEINAS.keys())):
            filas_col = sorted(
                f for c, f in MAPA_PROTEINAS.keys()
                if c == col and (c, f) not in coords_ancladas
            )
            for i in range(len(filas_col) - 1):
                f1, f2 = filas_col[i], filas_col[i + 1]
                if f2 != f1 + 1:
                    continue
                k1, k2 = (col, f1), (col, f2)
                # Los duplicados ARY009 son pares (1,2)=Bad, (3,4)=Bax, ...
                # Recorrer filas consecutivas genera tambien pares solapados
                # como (2,3), que son proteinas DISTINTAS: esas no se deduplican,
                # revertirlas destruiria una posicion legitima.
                if MAPA_PROTEINAS.get(k1) != MAPA_PROTEINAS.get(k2):
                    continue
                if k1 not in posiciones or k2 not in posiciones:
                    continue
                px1, py1 = posiciones[k1]
                px2, py2 = posiciones[k2]
                if abs(px1 - px2) <= 2 and abs(py1 - py2) <= 2:
                    gx1, gy1 = geo_posiciones[k1]
                    gx2, gy2 = geo_posiciones[k2]
                    d1 = (px1 - gx1) ** 2 + (py1 - gy1) ** 2
                    d2 = (px2 - gx2) ** 2 + (py2 - gy2) ** 2
                    if d1 >= d2:
                        posiciones[k1] = (gx1, gy1)
                        snapeadas.discard(k1)
                    else:
                        posiciones[k2] = (gx2, gy2)
                        snapeadas.discard(k2)

        # ── PASADA 4: AUTO-CORRECCIÓN por columna/fila usando spots snapeados ──
        # IDEA: las posiciones que SÍ se snapearon nos dicen dónde está REALMENTE
        # cada columna y fila en esta imagen. Las que NO se snapearon (porque el
        # spot es invisible o muy débil) deben usar ese mismo X / Y para no
        # quedar desalineadas del resto.
        x_mediano_col: Dict[str, int] = {}
        if snapeadas:
            # X mediano por columna (solo usando posiciones snapeadas)
            from collections import defaultdict
            x_por_col = defaultdict(list)
            y_por_fila = defaultdict(list)
            for k in snapeadas:
                col, fila = k
                px, py = posiciones[k]
                x_por_col[col].append(px)
                y_por_fila[fila].append(py)

            x_mediano_col = {col: int(round(float(np.median(xs))))
                             for col, xs in x_por_col.items() if len(xs) >= 3}
            y_mediano_fila = {fila: int(round(float(np.median(ys))))
                              for fila, ys in y_por_fila.items() if len(ys) >= 2}

            # Aplicar correcciones a posiciones NO snapeadas (espacios "invisibles")
            for k, (px, py) in list(posiciones.items()):
                if k in coords_ancladas or k in snapeadas:
                    continue  # no tocar referencias ni posiciones bien snapeadas
                col, fila = k
                nuevo_px, nuevo_py = px, py
                if col in x_mediano_col:
                    nuevo_px = x_mediano_col[col]
                if fila in y_mediano_fila:
                    nuevo_py = y_mediano_fila[fila]
                posiciones[k] = (nuevo_px, nuevo_py)

            # Corrección de posiciones snapeadas que se desviaron demasiado del
            # X mediano de su columna: probable snap a blob incorrecto en X.
            # Solo corrige outliers claros (> 25% del col_spacing).
            umbral_x_outlier = max(3, int(col_spacing * 0.25))
            for k, (px, py) in list(posiciones.items()):
                if k in coords_ancladas:
                    continue
                col, fila = k
                if col not in x_mediano_col:
                    continue
                if abs(px - x_mediano_col[col]) > umbral_x_outlier:
                    posiciones[k] = (x_mediano_col[col], py)
                    snapeadas.discard(k)

        # ── PASADA 5: Centrado fino ULTRA-CONSERVADOR ──
        # Para cada posición YA snapeada a un spot real, hacer un micro-ajuste
        # de máximo 2 píxeles para centrar exactamente en el píxel más brillante.
        # Esto corrige los casos donde el centroide del blob detectado está
        # ligeramente off-center respecto al pico real del spot.
        h_img, w_img = inv.shape
        for k in list(snapeadas):
            if k in coords_ancladas:
                continue
            px, py = posiciones[k]
            # ROI MUY pequeño: ±2 píxeles
            win = 2
            x0, x1 = max(0, px - win), min(w_img, px + win + 1)
            y0, y1 = max(0, py - win), min(h_img, py + win + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = inv[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            max_val = float(roi.max())
            current_val = float(inv[py, px])
            # Solo refinar si hay un máximo CLARAMENTE mejor que la posición actual
            if max_val < 15:  # contraste muy bajo
                continue
            if current_val >= max_val * 0.95:  # ya está prácticamente en el pico
                continue
            # Encontrar el centroide del pico (píxeles con valor >= 90% del max)
            mask = roi >= max_val * 0.90
            if mask.sum() < 1:
                continue
            ys_local, xs_local = np.where(mask)
            new_px = x0 + int(round(np.mean(xs_local)))
            new_py = y0 + int(round(np.mean(ys_local)))
            # Limitar movimiento a 2px máximo (super conservador)
            if abs(new_px - px) <= 2 and abs(new_py - py) <= 2:
                posiciones[k] = (new_px, new_py)

        # ── PASADA 6: SNAP SUAVE FINAL (±3 px) ──
        # Para CADA posición no anclada, hacer un último ajuste si hay un spot
        # detectado muy cerca (≤3 px). Esto corrige distorsiones LOCALES que
        # el snap normal o la auto-corrección no captaron. Mejora el centrado
        # visual sin afectar la lógica global.
        if spots_detectados:
            spots_xy_arr = np.array(
                [[s[0], s[1]] for s in spots_detectados], dtype=np.float32
            )
            snap_radio = 3
            snap_radio_sq = snap_radio * snap_radio
            for k, (px, py) in list(posiciones.items()):
                if k in coords_ancladas:
                    continue
                diffs = spots_xy_arr - np.array([px, py], dtype=np.float32)
                dist_sq = (diffs ** 2).sum(axis=1)
                idx_min = int(np.argmin(dist_sq))
                if dist_sq[idx_min] <= snap_radio_sq:
                    new_px = int(round(float(spots_xy_arr[idx_min, 0])))
                    new_py = int(round(float(spots_xy_arr[idx_min, 1])))
                    posiciones[k] = (new_px, new_py)

        # Las referencias y PBS se mantienen en sus posiciones snap originales
        for coord, pos in pos_ancladas.items():
            posiciones[coord] = pos

        return posiciones, inv, r

    # ------------------------------------------------------------------
    def _refinar_centrado_local(self, inv: np.ndarray, px: int, py: int, r: int) -> Tuple[int, int]:
        """
        Refinamiento fino por mean-shift local con ROI ajustado.

        Estrategia:
          1. ROI ligeramente más pequeño que la caja para evitar spots vecinos
          2. Encuentra el píxel MÁS brillante del ROI (centro probable del spot)
          3. Aplica umbral del 70% del máximo para usar solo el núcleo del spot
          4. Mean-shift con límite de movimiento por iteración

        Args:
            inv: imagen invertida (spots brillantes sobre fondo oscuro)
            px, py: posición actual (ya cerca del spot)
            r: radio de la caja
        Returns:
            (cx, cy) refinados al centro del spot
        """
        h, w = inv.shape
        cx, cy = float(px), float(py)
        # ROI más estricto: 85% del radio para evitar spots vecinos
        r_roi = max(3, int(r * 0.85))
        max_movimiento_por_iter = r * 0.5  # límite estricto para evitar saltos a vecinos

        for _ in range(4):
            x0, x1 = max(0, int(cx - r_roi)), min(w, int(cx + r_roi + 1))
            y0, y1 = max(0, int(cy - r_roi)), min(h, int(cy + r_roi + 1))
            if x1 <= x0 or y1 <= y0:
                break

            roi = inv[y0:y1, x0:x1].astype(np.float32)
            if roi.size == 0:
                break

            roi_max = float(roi.max())
            if roi_max < 8:  # spot muy débil/inexistente — no refinar
                break

            # Umbral del 70%: solo el núcleo brillante del spot
            # (más estricto que antes para no capturar penumbra de spots vecinos)
            umbral = roi_max * 0.70

            mask = roi > umbral
            if mask.sum() < 3:
                break

            # Pesos: intensidad sobre el umbral (lineal)
            pesos = (roi - umbral) * mask
            total = pesos.sum()
            if total < 1e-6:
                break

            ys, xs = np.mgrid[y0:y1, x0:x1]
            nueva_cx = (xs.astype(np.float32) * pesos).sum() / total
            nueva_cy = (ys.astype(np.float32) * pesos).sum() / total

            # Limitar movimiento por iteración
            dx = float(nueva_cx) - cx
            dy = float(nueva_cy) - cy
            if abs(dx) > max_movimiento_por_iter:
                dx = max_movimiento_por_iter * (1 if dx > 0 else -1)
            if abs(dy) > max_movimiento_por_iter:
                dy = max_movimiento_por_iter * (1 if dy > 0 else -1)

            cx += dx
            cy += dy

            # Convergencia
            if abs(dx) < 0.4 and abs(dy) < 0.4:
                break

        return int(round(cx)), int(round(cy))

    # ------------------------------------------------------------------
    @staticmethod
    def _promediar_duplicados(
        acum: Dict[str, List[float]],
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Promedia los duplicados de cada proteína.
        Cuando los dos duplicados difieren más de 2.5× y el mayor supera 15,
        se sigue usando el PROMEDIO (como indica el kit) pero la proteína se
        devuelve en la lista de discordantes para marcarla como baja confianza.

        Returns: (intensidades, lista_de_proteinas_discordantes)
        """
        resultado: Dict[str, float] = {}
        discordantes: List[str] = []
        for proteina, valores in acum.items():
            # Los controles se promedian SIEMPRE. En PBS (control negativo)
            # quedarse con el maximo invierte su sentido, y pbs_val se resta
            # a las 35 proteinas en cuantificar(): un PBS inflado recorta
            # todo el strip a la vez.
            if len(valores) == 2 and proteina not in ("PBS", "Reference"):
                v_max, v_min = max(valores), min(valores)
                if v_max >= 15 and v_min < v_max * 0.40:
                    # Antes se usaba el máximo, lo que sesga hacia arriba.
                    # Ahora se promedia (método del kit) y se MARCA como baja
                    # confianza: tanta diferencia suele indicar grilla corrida.
                    discordantes.append(proteina)
                    logger.warning(
                        f"Duplicados discordantes [{proteina}]: "
                        f"{v_min:.1f} vs {v_max:.1f} → promedio, BAJA CONFIANZA"
                    )
            resultado[proteina] = round(float(np.mean(valores)), 3)
        return resultado, discordantes

    # ------------------------------------------------------------------
    def _muestrear_y_posiciones(
        self,
        strip_gris: np.ndarray,
        tl: Tuple,
        tr: Tuple,
        bl: Tuple,
        pbs_bot: Optional[Tuple] = None,
        img_muestreo: Optional[np.ndarray] = None,
        mask_sat: Optional[np.ndarray] = None,
    ):
        """
        Las POSICIONES se calculan sobre strip_gris (8 bits, donde están
        calibrados los detectores). Las INTENSIDADES se leen de img_muestreo
        (16 bits si existe) y se expresan en escala 0-255 con decimales, para
        no perder precisión sin cambiar los umbrales del resto del pipeline.
        Deja en self._ultimas_alertas las proteínas con duplicados
        discordantes y las que tienen píxeles saturados.

        Igual que muestrear_strip pero devuelve también (posiciones, r)
        para que el llamador pueda pasarlas a _debug_strip sin reejecutar
        el pipeline.
        Returns: (intensidades, posiciones, r)
        """
        h, w = strip_gris.shape
        posiciones, inv, r = self._calcular_posiciones_finales(
            strip_gris, tl, tr, bl, pbs_bot)
        # El paso de fila acota la ventana del centroide (ver _centrado_fino).
        # Se calcula igual que en _limitar_a_su_celda, desde tl/bl.
        sep_y = abs(bl[1] - tl[1])
        paso_fila = (sep_y / (self._FILA_REF_BOT - self._FILA_REF_TOP)
                     if sep_y >= 1 else 0.0)
        # `pos_float` recoge la posición subpixel del mean-shift para muestrear
        # la intensidad sin el error del redondeo (ver el muestreo más abajo).
        # Pasa por el cinturón de celda igual que la entera.
        pos_float: Dict[Tuple[str, int], Tuple[float, float]] = {}
        posiciones = self._centrado_fino(strip_gris, posiciones, r, paso_fila,
                                         pos_float=pos_float)
        posiciones = self._limitar_a_su_celda(posiciones, tl, tr, bl,
                                              pos_float=pos_float)

        inv_m = inv
        if img_muestreo is not None and img_muestreo.shape == strip_gris.shape:
            if img_muestreo.dtype == np.uint16:
                claro16 = float(np.mean(img_muestreo)) > 65535 * 0.35
                base16 = cv2.bitwise_not(img_muestreo) if claro16 else img_muestreo
                inv_m = base16.astype(np.float32) / 256.0

        # El parche de intensidad se toma con precisión SUBPIXEL, centrado en
        # `pos_float` (el centroide exacto al que convergió el mean-shift) y no
        # en la posición entera. Motivo: el recorte entero
        # inv_m[py-r:py+r, px-r:px+r] lee la mancha desde una posición
        # redondeada, y con r = 3-4 px y manchas de 2-3 px reales ese redondeo
        # mete hasta 0.71 px de error en diagonal — del mismo orden que el
        # residuo de centrado que todavía queda (mediana 0.117·r medida), o sea
        # es una de las fuentes dominantes de error que sobreviven al ciclo 4.
        # Golpea de lleno a la diferencia entre DUPLICADOS: los dos duplicados
        # de una proteína son filas consecutivas, cada uno se redondea por su
        # cuenta, y 1 px de diferencia entre los dos recortes alcanza para que
        # lean distinto aunque la proteína sea la misma y las dos manchas
        # idénticas. Medido sobre las 6 membranas: la mediana de la diferencia
        # entre duplicados baja 6.90% -> 6.17% (-10.7% relativo).
        #
        # El centro se pide en (fx-0.5, fy-0.5) a propósito: `getRectSubPix`
        # centra el parche en su centro GEOMÉTRICO, mientras que el recorte
        # [px-r, px+r) de toda la vida está centrado en px-0.5. Pasar (fx, fy)
        # "a secas" corre las 1656 ventanas medio píxel respecto de la
        # convención con la que están calibrados los umbrales del pipeline
        # (>8, >10, >15) y el fondo. Se midió: baja un poco más la diferencia
        # entre duplicados (6.17% -> 5.95%) pero SUBE la contaminación de las
        # celdas vacías en bloque (mediana 1.076 -> 1.148, media +0.103, p90
        # 4.39 -> 4.54, y de 13 a 18 celdas vacías leyendo >10). Con el medio
        # píxel compensado la distribución de las celdas vacías queda igual
        # (mediana 1.076 idéntica, media -0.011, mismos recuentos >5 y >8).
        # NO quitar el -0.5 sin recalibrar esos umbrales.
        #
        # `getRectSubPix` exige float32: en uint8 redondearía la interpolación
        # bilineal a entero y volvería a perder justo el subpíxel que se busca.
        # Si el parche no entra completo en el strip se cae al recorte entero de
        # antes (getRectSubPix no puede interpolar fuera del borde). Medido:
        # 1790 cuadros por la vía subpixel contra 4 por el borde (0.2%).
        inv_sub = (inv_m if inv_m.dtype == np.float32
                   else inv_m.astype(np.float32))
        lado = 2 * r
        acum: Dict[str, List[float]] = {}
        saturadas: List[str] = []
        for (col, fila), proteina in MAPA_PROTEINAS.items():
            px, py = posiciones[(col, fila)]
            x0, x1 = max(0, px - r), min(w, px + r)
            y0, y1 = max(0, py - r), min(h, py + r)
            fx, fy = pos_float.get((col, fila), (float(px), float(py)))
            if (r + 1 <= fx <= w - r - 1) and (r + 1 <= fy <= h - r - 1):
                valor = float(cv2.getRectSubPix(
                    inv_sub, (lado, lado), (fx - 0.5, fy - 0.5)).mean())
            else:
                roi = inv_m[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                valor = float(roi.mean())
            acum.setdefault(proteina, []).append(valor)
            if (mask_sat is not None and mask_sat.shape == strip_gris.shape
                    and mask_sat[y0:y1, x0:x1].any() and proteina not in saturadas):
                saturadas.append(proteina)

        intensidades, discordantes = self._promediar_duplicados(acum)
        intensidades = {k: round(v, 3) for k, v in intensidades.items()}
        for prot in saturadas:
            if prot == 'Reference':
                logger.warning("Reference Spot SATURADO: la normalización de "
                               "esta tirilla no es confiable.")
            else:
                logger.warning(f"Spot saturado [{prot}]: su intensidad real "
                               f"puede ser mayor a la medida.")
        self._ultimas_alertas = {'discordantes': discordantes,
                                 'saturadas': saturadas}
        return intensidades, posiciones, r

    def _limitar_a_su_celda(
        self,
        posiciones: Dict[Tuple[str, int], Tuple[int, int]],
        tl: Tuple, tr: Tuple, bl: Tuple,
        fraccion: float = 0.30,
        pos_float: Optional[Dict[Tuple[str, int], Tuple[float, float]]] = None,
    ) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """
        Cinturón de seguridad: ningún cuadro puede quedar a más de `fraccion`
        del paso de columna/fila respecto de su posición geométrica teórica.
        Más allá de medio paso el cuadro está sobre la proteína VECINA: los
        ajustes por spot (snap, ICP, centrado fino) lo atraen hacia una mancha
        cercana cuando la proteína propia no se expresa, y entonces se mide la
        señal de otra proteína. Este límite lo impide.

        Si se pasa `pos_float` (las posiciones subpixel de `_centrado_fino`),
        se recorta IN SITU con la misma caja. Tiene que recortarse: el
        muestreo de intensidad usa el float, así que un float sin cinturón se
        escaparía del límite que la posición entera sí respeta y volvería a
        leer la proteína vecina por la puerta de atrás.
        """
        sep_x = abs(tr[0] - tl[0])
        sep_y = abs(bl[1] - tl[1])
        if sep_x < 1 or sep_y < 1:
            return posiciones
        paso_x = sep_x * min(abs(self._T_COLS['B'] - self._T_COLS['A']),
                             abs(self._T_COLS['C'] - self._T_COLS['B']),
                             abs(self._T_COLS['D'] - self._T_COLS['C']))
        span = self._FILA_REF_BOT - self._FILA_REF_TOP
        paso_y = sep_y / span
        lim_x, lim_y = fraccion * paso_x, fraccion * paso_y
        nuevas = {}
        n_lim = 0
        for (col, fila), (px, py) in posiciones.items():
            t_row = (fila - self._FILA_REF_TOP) / span
            x_a = tl[0] + t_row * (bl[0] - tl[0])
            gx = x_a + self._T_COLS[col] * (tr[0] - tl[0])
            gy = tl[1] + t_row * (bl[1] - tl[1])
            nx = min(max(px, gx - lim_x), gx + lim_x)
            ny = min(max(py, gy - lim_y), gy + lim_y)
            if pos_float is not None and (col, fila) in pos_float:
                fx, fy = pos_float[(col, fila)]
                pos_float[(col, fila)] = (
                    min(max(fx, gx - lim_x), gx + lim_x),
                    min(max(fy, gy - lim_y), gy + lim_y))
            nx, ny = int(round(nx)), int(round(ny))
            if (nx, ny) != (px, py):
                n_lim += 1
            nuevas[(col, fila)] = (nx, ny)
        if n_lim:
            logger.info(f"Límite de celda: {n_lim} cuadros devueltos hacia su "
                        f"posición (estaban sobre la proteína vecina).")
        return nuevas

    def _centrado_fino(
        self,
        strip_gris: np.ndarray,
        posiciones: Dict[Tuple[str, int], Tuple[int, int]],
        r: int,
        paso_fila: float = 0.0,
        pasadas: int = 3,
        pos_float: Optional[Dict[Tuple[str, int], Tuple[float, float]]] = None,
    ) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """
        Último ajuste por cuadro: mueve cada cuadro al centroide de intensidad
        de SU mancha. En imágenes ChemiDoc (348x260) cada spot mide 2-3 px
        reales, así que 1 px de error deja el spot en el borde del cuadro y
        subestima su intensidad (hasta -70 % medido en Sw620).

        `paso_fila` es la separación entre filas en px; acota la ventana de
        búsqueda del centroide (ver abajo). Si no se pasa, se mantiene el
        comportamiento viejo.

        Salvaguardas:
          - Solo se mueve si hay una mancha con contraste claro (>= 12 en 8 bits).
          - Desplazamiento máximo 2.0·r, solo como tope de la trayectoria del
            mean-shift: el que impide saltar al spot vecino es la caja de
            `_limitar_a_su_celda` (ver el techo más abajo).
          - Solo se consideran píxeles >= 50 % del pico (ignora fondo y halos).
        Los cuadros sin mancha (proteína no expresada) no se tocan.

        `pasadas` es el número de iteraciones del mean-shift (ver el bucle).
        Con 1 el comportamiento es el de una sola medición de centroide.

        `pos_float` es un diccionario de SALIDA opcional: si se pasa, queda con
        la posición FLOAT exacta a la que convergió el mean-shift de cada
        cuadro, antes de redondear. Lo que se DEVUELVE sigue siendo entero
        porque el Mapa Visual, `_debug_strip` y las métricas dibujan y miden
        con enteros; pero el muestreo de intensidad necesita el float (ver
        `_muestrear_y_posiciones`): redondear la posición antes de leer la
        mancha mete hasta 0.71 px de error en diagonal, y con manchas de 2-3 px
        reales eso es del mismo orden que el residuo de centrado que todavía
        queda (0.117·r medido), o sea el redondeo ya es una de las fuentes
        dominantes de error de intensidad.
        """
        if r is None or r < 1:
            return posiciones
        img = cv2.GaussianBlur(strip_gris, (3, 3), 0)
        if img.dtype != np.uint8:
            img = (img / 256).astype(np.uint8)
        claro = float(np.mean(img)) > 255 * 0.35
        señal = (255 - img.astype(np.float32)) if claro else img.astype(np.float32)
        h, w = señal.shape
        # Media-ventana de búsqueda del centroide. NO puede llegar a media fila:
        # si llega, la ventana contiene también el spot de la fila vecina y el
        # centroide cae en el punto medio entre las dos manchas en vez del centro
        # de la propia. Con r = 0.40 del paso de fila (get_radio_dinamico), el
        # viejo 1.6*r era 0.64 del paso: la ventana (lado 11-13 px) superaba el
        # paso de fila (9.4-12.4 px) en las 6 membranas de prueba, SIEMPRE
        # alcanzaba a la vecina. Por eso el centrado movía 1 de cada 3 cuadros y
        # apenas bajaba el descentrado (29.9% -> 28.2%), empeorando la cola
        # (p90 0.810 -> 0.926). Acotado a 0.40 del paso queda en 3-4 px, debajo
        # de medio paso y por encima de la semi-extensión medida de una mancha
        # (mediana 3 px, p90 4 px), así que la mancha propia entra completa y la
        # vecina no. Medido: 25.0% -> 18.2% de cuadros descentrados, p90 0.621.
        # La cota es el paso de FILA en los dos ejes aunque el paso de columna
        # sea mayor: el cuadro de muestreo es cuadrado (px±r, py±r), así el
        # centroide se mide sobre los mismos píxeles que después se promedian.
        # Ensanchar la ventana en x hasta lo que permite el paso de columna se
        # midió y empeora (19.5% de descentrados).
        R = int(round(r * 1.6))
        if paso_fila > 0:
            R = max(2, min(R, int(paso_fila * 0.40)))
        # Techo del desplazamiento. El 0.6·r viejo recortaba el 17.2% de los
        # movimientos del centroide en las 6 membranas de prueba: el cuadro
        # "apuntaba" al centro de su mancha pero se quedaba a mitad de camino,
        # y ese resto de descentrado subestima la intensidad igual que si no se
        # hubiera centrado. Subirlo a 1.0·r NO puede alcanzar la mancha vecina:
        #   - la ventana del centroide quedó acotada en R = r (medido: R/r = 1.00
        #     en las 23 tirillas), así que el centroide no puede pedir más de
        #     ~1·r salvo en la diagonal (máximo medido 1.32·r);
        #   - medio paso de fila son 1.41·r en el peor caso (mediana 1.55·r),
        #     o sea el spot vecino queda fuera de alcance;
        #   - y después corre `_limitar_a_su_celda`, cuyo ±0.30 del paso equivale
        #     a 0.85-0.99·r: es un cinturón MÁS estrecho que este techo.
        # Por eso al subirlo bajan las cinco métricas a la vez en vez de subir:
        # descentrado 18.2% -> 16.0%, p90 0.621 -> 0.561, superpuestos 1 -> 0,
        # dif. entre duplicados 8.0% -> 7.7%, sobre_vecina 0 -> 0.
        #
        # 1.0·r era el techo hasta que el mean-shift pasó a 3 pasadas, y ahí se
        # volvió la compuerta que FRENA el centrado. Dos motivos medidos:
        #   - el techo es un CÍRCULO de radio max_shift, pero `_limitar_a_su_celda`
        #     es una CAJA de medios lados ±lim_x, ±lim_y (0.93-1.14·r y 0.85-0.99·r
        #     en las 6 membranas). La diagonal de esa caja llega a 1.51·r, o sea
        #     el círculo de 1.0·r le corta las ESQUINAS: un cuadro que necesita
        #     moverse en diagonal queda topado por el techo aunque la caja (el
        #     guardián con sentido geométrico) todavía se lo permita.
        #   - el techo se aplica en cada pasada, así que además congela la ventana
        #     del mean-shift fuera de la mancha y el cuadro ya no converge.
        # Medido con 3 pasadas: el 13.9% de los cuadros con mancha quedaban
        # topados, y eran el 59% de los que terminaban descentrados.
        # Barrido del techo (descentrado / p90 del offset), cinturón intacto:
        #   1.0 -> 11.5% / 0.435   1.2 -> 10.2% / 0.367   1.5 -> 9.8% / 0.345
        #   1.6 ->  9.7% / 0.343   2.0 ->  9.6% / 0.342   3.0 y sin techo -> idem
        # 2.0 es el punto de SATURACIÓN (2.0, 3.0 y sin techo dan las cinco
        # métricas idénticas), así que no es un valor ajustado al dato: es
        # simplemente "lo bastante grande para no ser el que manda", y cubre la
        # diagonal máxima de la caja (1.51·r) con margen para la trayectoria.
        # SUBIR ESTE TECHO NO PUEDE ACERCAR UN CUADRO A LA PROTEÍNA VECINA: el
        # que fija el peor caso es la caja de `_limitar_a_su_celda`, que corre
        # DESPUÉS y recorta toda posición final a ±0.30 del paso respecto de la
        # geometría. Con eso, dos cuadros vecinos nunca quedan a menos de
        # 0.40·paso (= 1.13-1.32·r) y ninguno puede pasar del 0.47·paso que
        # dispara `sobre_vecina`, sea cual sea este techo. Medido: superpuestos
        # 0 -> 0, sobre_vecina 0 -> 0, y los cuadros cuya celda está vacía en la
        # posición geométrica y terminan leyendo señal (>10) bajan 14 -> 13, con
        # la contaminación mediana de las celdas vacías igual (1.08).
        # Para revertir: volver a `r * 1.0`.
        max_shift = r * 2.0
        nuevas: Dict[Tuple[str, int], Tuple[int, int]] = {}
        n_mov = 0
        n_pas = 0
        for clave, (px, py) in posiciones.items():
            # Mean-shift: el centroide se RECALCULA desde la posición ya
            # corregida, no una sola vez desde la grilla. Motivo: la ventana
            # está acotada en R = r (arriba) para no ver la mancha vecina, y
            # una mancha más ANCHA que 2r entra recortada en esa ventana, con
            # lo cual el centroide queda sesgado HACIA el centro viejo (sesgo
            # de truncamiento): el cuadro apunta bien pero se queda corto.
            # Medido en el ciclo 1: de los cuadros que EMPEORARON su offset,
            # 28/28 ya estaban dentro de su propia celda, estaban sobre manchas
            # fuertes y anchas (intensidad mediana 31.7 contra 20.7 de los que
            # mejoraron) y se movieron hacia la geometría mientras el offset
            # subía 0.29 -> 0.63; solo 5 de 28 topaban contra el cinturón de
            # celda, así que el culpable era la ventana y no el cinturón.
            # ENSANCHAR la ventana no sirve (se midió: acotar x por el paso de
            # columna da 19.5% de descentrados contra 18.2%), porque alcanza a
            # la vecina. Iterar sí: cada pasada centra más la ventana chica
            # sobre la mancha propia, el recorte se vuelve simétrico y la
            # ventana nunca llega a ver a la vecina.
            # Medido con 1/2/3/5 pasadas (descentrado, dif. duplicados):
            # 16.0%/7.7% -> 11.8%/6.9% -> 11.5%/6.8% -> 11.5%/6.8%. O sea
            # SATURA en 3 (con 5 sale idéntico), y por eso el default es 3.
            # El grupo que este cambio tenía que arreglar es el que más baja:
            # manchas fuertes (intensidad > 25) 18.9% -> 12.1% de descentrados,
            # p90 del offset 0.612 -> 0.432. Superpuestos y sobre_vecina siguen
            # en 0 y las tirillas en 23/24. Con `pasadas=1` el resultado es
            # exactamente el de una sola medición (verificado), así que bajarlo
            # a 1 es la forma de revertir esto sin tocar el código.
            cx, cy = float(px), float(py)
            for _ in range(max(1, pasadas)):
                ix, iy = int(round(cx)), int(round(cy))
                x0, x1 = max(0, ix - R), min(w, ix + R + 1)
                y0, y1 = max(0, iy - R), min(h, iy + R + 1)
                roi = señal[y0:y1, x0:x1]
                if roi.size == 0:
                    break
                roi = roi - np.percentile(roi, 40)
                roi[roi < 0] = 0
                pico = float(roi.max())
                if pico < 12:
                    break
                roi[roi < 0.5 * pico] = 0
                ys, xs = np.mgrid[y0:y1, x0:x1]
                tot = float(roi.sum())
                ncx = float((xs * roi).sum()) / tot
                ncy = float((ys * roi).sum()) / tot
                # El techo es ACUMULADO desde la posición de partida, no por
                # pasada: si no, 3 pasadas de 1.0·r sumarían 3·r y el cuadro
                # llegaría a la mancha vecina (medio paso de fila son 1.41·r
                # en el peor caso).
                tdx, tdy = ncx - px, ncy - py
                td = (tdx * tdx + tdy * tdy) ** 0.5
                if td > max_shift:
                    ncx = px + tdx * max_shift / td
                    ncy = py + tdy * max_shift / td
                paso = ((ncx - cx) ** 2 + (ncy - cy) ** 2) ** 0.5
                cx, cy = ncx, ncy
                n_pas += 1
                if paso < 0.1:
                    break   # convergió: más pasadas no mueven nada
            nx, ny = int(round(cx)), int(round(cy))
            if (nx, ny) != (px, py):
                n_mov += 1
            nuevas[clave] = (nx, ny)
            if pos_float is not None:
                pos_float[clave] = (cx, cy)
        # Anti-colapso: si el centrado deja dos cuadros encimados (ambos
        # atraídos por la MISMA mancha), los dos vuelven a su posición de
        # grilla. Medir dos proteínas sobre un mismo spot es peor que un
        # cuadro ligeramente descentrado.
        claves = list(nuevas.keys())
        lim_sq = (1.2 * r) ** 2
        revertir = set()
        for i in range(len(claves)):
            xi, yi = nuevas[claves[i]]
            for j in range(i + 1, len(claves)):
                xj, yj = nuevas[claves[j]]
                if (xi - xj) ** 2 + (yi - yj) ** 2 < lim_sq:
                    oi, oj = posiciones[claves[i]], posiciones[claves[j]]
                    if (oi[0] - oj[0]) ** 2 + (oi[1] - oj[1]) ** 2 >= lim_sq:
                        revertir.update((claves[i], claves[j]))
        for c in revertir:
            nuevas[c] = posiciones[c]
            # El float vuelve junto con el entero: si no, el anti-colapso
            # dejaría el cuadro dibujado en su posición de grilla pero midiendo
            # todavía sobre la mancha compartida que disparó la colisión.
            if pos_float is not None:
                pos_float[c] = (float(posiciones[c][0]),
                                float(posiciones[c][1]))
        logger.info(f"Centrado fino: {n_mov}/{len(posiciones)} cuadros ajustados "
                    f"({n_pas} pasadas de mean-shift), "
                    f"{len(revertir)} revertidos por colisión.")
        return nuevas

    def muestrear_strip(
        self,
        strip_gris: np.ndarray,
        tl: Tuple,
        tr: Tuple,
        bl: Tuple,
        pbs_bot: Optional[Tuple] = None,
    ) -> Dict[str, float]:
        """
        Muestrea intensidad en cada posición del mapa ARY009.
        Usa _calcular_posiciones_finales() para el pipeline de posicionamiento
        y luego promedia los duplicados de cada proteína.
        """
        h, w = strip_gris.shape
        posiciones, inv, r = self._calcular_posiciones_finales(
            strip_gris, tl, tr, bl, pbs_bot)

        acum: Dict[str, List[float]] = {}
        for (col, fila), proteina in MAPA_PROTEINAS.items():
            px, py = posiciones[(col, fila)]
            x0, x1 = max(0, px - r), min(w, px + r)
            y0, y1 = max(0, py - r), min(h, py + r)
            roi = inv[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            acum.setdefault(proteina, []).append(float(roi.mean()))

        intensidades, _ = self._promediar_duplicados(acum)
        return intensidades

    # ------------------------------------------------------------------
    def analizar_imagen(
        self,
        ruta: str,
        n_strips: int = 4,
        ruta_debug: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """
        Pipeline completo para una imagen con N membranas.

        Returns: Lista de dicts {proteina: intensidad_bruta}, uno por strip.
        """
        # Cargar imagen preservando profundidad de bits original (8 ó 16-bit)
        gris_orig, gris_8 = self._cargar_imagen(ruta)

        bit_depth = 16 if gris_orig.dtype == np.uint16 else 8
        logger.info(
            f"Imagen cargada: {gris_orig.shape[1]}x{gris_orig.shape[0]}, {bit_depth}-bit"
            f"{' (precisión extendida)' if bit_depth == 16 else ''}"
        )

        # Separar membranas sobre la versión 8-bit (los umbrales del Black-Hat,
        # CLAHE y proyección están afinados para 0-255). El muestreo posterior
        # sí usa la profundidad original.
        imagen_suave = cv2.GaussianBlur(gris_8, (3, 3), 0)
        strips = self.separar_strips(imagen_suave, n_strips)

        if not strips:
            raise ValueError(
                f"No se detectaron tirillas en la imagen. "
                f"Verifique que la imagen contenga {n_strips} membranas visibles "
                f"(formato esperado: fondo claro, manchas oscuras, o imagen ChemiDoc)."
            )

        if len(strips) < n_strips:
            logger.warning(
                f"Solo se detectaron {len(strips)}/{n_strips} tirillas. "
                f"El análisis continuará con las tirillas encontradas."
            )

        debug_img = None
        if ruta_debug:
            debug_img = cv2.cvtColor(gris_8, cv2.COLOR_GRAY2BGR)

        resultados: List[Dict[str, float]] = []
        self._strips_visual = []
        for i, (strip_8, (ox, oy, ow, oh)) in enumerate(strips):
            # Recortar la versión original (16-bit si aplica) para esta tirilla.
            # Se guarda para permitir re-muestreo con referencias manuales.
            strip_orig = gris_orig[oy:oy + oh, ox:ox + ow]
            strip_pos = gris_8[oy:oy + oh, ox:ox + ow]
            mask_sat = getattr(self, '_mask_sat', None)
            sat_strip = mask_sat[oy:oy + oh, ox:ox + ow] if mask_sat is not None else None

            refs = self.encontrar_referencias(strip_8)
            # Una tirilla con anclajes imposibles se trata igual que una sin
            # referencias: mejor no entregar intensidades que entregar números
            # calculados sobre un grid que no corresponde a la membrana.
            if refs is not None and not self._geometria_plausible(refs, i + 1):
                refs = None
            if refs is None:
                logger.warning(f"Strip {i+1}: sin referencias. Devolviendo intensidades vacías.")
                resultados.append({})
                self._strips_visual.append({
                    'strip_img': strip_8.copy(),
                    'strip_orig': strip_orig.copy(),
                    'refs': None,
                    'offset': (ox, oy, ow, oh),
                    'strip_pos': strip_pos.copy(),
                    'mask_sat': sat_strip,
                    'alertas': {'sin_referencias': True},
                })
                continue

            tl, tr, bl = refs

            intensidades, posiciones_debug, r_debug = self._muestrear_y_posiciones(
                strip_pos, tl, tr, bl, img_muestreo=strip_orig, mask_sat=sat_strip)
            resultados.append(intensidades)

            if debug_img is not None:
                # Pasar posiciones ya calculadas para no repetir el pipeline
                self._debug_strip(
                    debug_img, strip_8, tl, tr, bl, ox, oy, i + 1,
                    posiciones_cache=posiciones_debug, r_cache=r_debug,
                )

            self._strips_visual.append({
                'strip_img': strip_8.copy(),
                'strip_orig': strip_orig.copy(),
                'refs': (tl, tr, bl),
                'offset': (ox, oy, ow, oh),
                'strip_pos': strip_pos.copy(),
                'mask_sat': sat_strip,
                'alertas': dict(self._ultimas_alertas),
                # Posiciones calculadas por el pipeline completo (cache para mapa visual)
                'posiciones_cache': posiciones_debug,
                'r_cache': r_debug,
            })

        if ruta_debug and debug_img is not None:
            cv2.imwrite(ruta_debug, debug_img)
            logger.info(f"Debug guardado: {ruta_debug}")

        return resultados

    # ------------------------------------------------------------------
    def _cargar_imagen(self, ruta: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga imagen preservando la profundidad de bits original (8 ó 16-bit).

        Returns:
            (gris_orig, gris_8bit): la primera mantiene la profundidad original
            para muestreo de intensidad de alta precisión; la segunda es 8-bit
            para los pasos de detección/separación.
        """
        ext = os.path.splitext(ruta)[1].lower()
        wsi_exts = {'.scn', '.svs', '.ndpi', '.mrxs', '.vms', '.bif'}

        # Los .scn de Bio-Rad ChemiDoc NO son WSI de Leica: openslide no los
        # abre. Se leen con el lector propio, conservando los 16 bits.
        es_biorad = False
        if ext == '.scn':
            with open(ruta, 'rb') as _f:
                es_biorad = b'Image Lab' in _f.read(300)
        if es_biorad:
            from convertir_scn import leer_scn_biorad
            img = leer_scn_biorad(ruta)
        elif ext in wsi_exts:
            try:
                import openslide
                slide = openslide.OpenSlide(ruta)
                dim = slide.dimensions
                escala = 4096 / max(dim)
                thumb = slide.get_thumbnail((int(dim[0] * escala), int(dim[1] * escala)))
                gris = cv2.cvtColor(np.array(thumb.convert('RGB')), cv2.COLOR_RGB2GRAY)
                slide.close()
                return gris, gris  # openslide entrega siempre 8-bit
            except ImportError:
                raise ImportError("Instalar openslide-python para archivos WSI (.scn, .svs, etc.)")

        else:
            # IMREAD_UNCHANGED preserva 16-bit y canales originales
            img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar: {ruta}")

        # Quitar alfa si lo tiene
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        # A escala de grises preservando dtype
        if img.ndim == 3:
            # cv2.cvtColor preserva uint16 al ir BGR→GRAY
            gris_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gris_orig = img

        # Versión 8-bit para detección
        if gris_orig.dtype == np.uint16:
            gris_8 = (gris_orig / 256).astype(np.uint8)
        elif gris_orig.dtype != np.uint8:
            # Por si llega float u otro tipo, normalizar a 8-bit
            gris_8 = cv2.normalize(gris_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gris_orig = gris_8
        else:
            gris_8 = gris_orig

        # ── AUTO-UPSCALE para imágenes pequeñas ──
        # El motor está calibrado para imágenes de ~1000+ px de altura.
        # Para imágenes pequeñas (como ChemiDoc directo: 348x260), upscalamos
        # con interpolación cúbica para que los kernels Black-Hat funcionen bien.
        h_img, w_img = gris_8.shape
        if max(h_img, w_img) < 800:
            factor = int(np.ceil(800 / max(h_img, w_img)))
            new_w = w_img * factor
            new_h = h_img * factor
            logger.info(
                f"Imagen pequeña ({w_img}x{h_img}) → upscale {factor}x → {new_w}x{new_h}"
            )
            gris_8 = cv2.resize(gris_8, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            if gris_orig.dtype == np.uint16:
                gris_orig = cv2.resize(gris_orig, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                gris_orig = gris_8.copy()

        # ── DETECCIÓN DE MODO: fondo claro vs oscuro ──
        # ARY009 estándar = fondo CLARO con spots OSCUROS.
        # ChemiDoc en modo Chemi produce el INVERSO: fondo OSCURO con
        # spots CLAROS (luminiscencia). Si detectamos modo oscuro, invertimos
        # Y normalizamos el fondo para emular el formato estándar.
        p80 = float(np.percentile(gris_8, 80))
        max_val = 255.0
        if p80 < max_val * 0.50:
            logger.info(
                f"Imagen en modo OSCURO detectada (p80={p80:.0f}<{max_val*0.5:.0f}). "
                "Invirtiendo + normalizando fondo a formato estándar."
            )
            # Inversión simple
            gris_8 = 255 - gris_8

            # LIMPIAR FONDO con threshold morfológico:
            # En la imagen invertida, el fondo tiene un "gradiente" gris suave
            # que confunde la detección de tirillas. Lo eliminamos haciendo
            # un threshold suave: cualquier pixel cerca del color de fondo
            # se lleva a 255 puro (blanco).

            # Identificar el fondo: la MODA de los píxeles claros
            hist = cv2.calcHist([gris_8], [0], None, [256], [0, 256])
            # Encontrar el pico del histograma en la zona clara (>= 128)
            pico_claro = int(128 + np.argmax(hist[128:]))
            # Si el pico del fondo está abajo de 250, llevarlo a 255
            if pico_claro < 250 and pico_claro > 100:
                # Saturar todo lo que sea más claro que (pico_claro - 15)
                # Esto lleva el fondo a blanco puro, manteniendo el contraste
                # de los spots oscuros.
                umbral_fondo = max(100, pico_claro - 15)
                mask_fondo = gris_8 >= umbral_fondo
                # fondo (>=umbral) → 255 puro; contenido (<umbral) → estirado a [0, 255]
                resultado = gris_8.astype(np.float32)
                resultado = np.where(mask_fondo, 255.0,
                                      np.clip(resultado / umbral_fondo * 255, 0, 255))
                gris_8 = resultado.astype(np.uint8)
                logger.info(f"Fondo blanqueado (pico={pico_claro}, umbral={umbral_fondo})")

            # Aplicar misma transformación a gris_orig
            if gris_orig.dtype == np.uint16:
                gris_orig = 65535 - gris_orig
            elif gris_orig.dtype == np.uint8:
                gris_orig = gris_8.copy()
            else:
                gris_orig = gris_8

        # Píxeles saturados (o casi): en formato fondo claro el detector
        # saturado queda en ~0. Un spot con estos píxeles subestima su
        # intensidad real, y si es un Reference distorsiona TODA la tirilla.
        umbral_sat = 256 if gris_orig.dtype == np.uint16 else 1
        self._mask_sat = gris_orig <= umbral_sat

        return gris_orig, gris_8

    # ------------------------------------------------------------------
    def cuantificar(self, intensidades: Dict[str, float]) -> Dict:
        """
        Aplica PBS-subtraction y normalización por Reference Spots.
        Devuelve la misma estructura que cuantificar_proteinas().
        """
        ref_val = intensidades.get('Reference', 0.0)
        pbs_val = intensidades.get('PBS', 0.0)
        ref_neta = max(EPSILON, ref_val - pbs_val)

        proteinas = {}
        for nombre, bruta in intensidades.items():
            if nombre in ('Reference', 'PBS'):
                continue
            neta = max(0.0, bruta - pbs_val)
            normalizada = round(neta / ref_neta, 6)
            tipo = ('pro-apoptótica' if nombre in PRO_APOPTOTICAS else
                    'anti-apoptótica' if nombre in ANTI_APOPTOTICAS else 'otra')
            proteinas[nombre] = {
                'intensidad_bruta': round(bruta, 2),
                'intensidad_neta': round(neta, 2),
                'normalizada': normalizada,
                'tipo': tipo,
            }

        return {
            'proteinas': proteinas,
            'pbs': pbs_val,
            'referencia': ref_val,
            'referencia_neta': round(ref_neta, 2),
        }

    # ------------------------------------------------------------------
    def _debug_strip(self, img, strip, tl, tr, bl, ox, oy, num,
                     posiciones_cache=None, r_cache=None):
        """
        Dibuja sobre la imagen debug las posiciones REALES del pipeline completo
        (incluyendo snap + consistencia + ICP).
        Si se pasan posiciones_cache y r_cache (ya calculados en muestrear_strip),
        los usa directamente para evitar re-ejecutar el pipeline.
        """
        if posiciones_cache is not None and r_cache is not None:
            posiciones = posiciones_cache
            r = r_cache
        else:
            # Calcular posiciones con el pipeline completo (caso fallback)
            posiciones, _inv, r = self._calcular_posiciones_finales(strip, tl, tr, bl)

        # Anchors de referencia
        for px, py, lbl in [(tl[0], tl[1], 'TL'), (tr[0], tr[1], 'TR'), (bl[0], bl[1], 'BL')]:
            cv2.circle(img, (int(ox + px), int(oy + py)), r + 4, (0, 140, 255), 2)
            cv2.putText(img, lbl, (int(ox + px) + r + 2, int(oy + py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        # Spots muestreados (posiciones reales del pipeline)
        for (col, fila), (px, py) in posiciones.items():
            proteina = MAPA_PROTEINAS.get((col, fila), '?')
            if proteina == 'Reference':
                color = (0, 215, 255)  # dorado
            elif proteina == 'PBS':
                color = (200, 180, 0)  # cian oscuro
            elif proteina in PRO_APOPTOTICAS:
                color = (0, 0, 200)    # rojo
            elif proteina in ANTI_APOPTOTICAS:
                color = (200, 0, 0)    # azul
            else:
                color = (0, 180, 0)    # verde

            cv2.rectangle(img,
                          (ox + px - r, oy + py - r),
                          (ox + px + r, oy + py + r), color, 1)

        cv2.putText(img, f"M{num}", (ox + 4, oy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


    # ------------------------------------------------------------------
    def recalibrar_strip(
        self,
        strip_idx: int,
        tl: Tuple[float, float],
        tr: Tuple[float, float],
        bl: Tuple[float, float],
        pbs_bot: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Recalibra un strip usando 3 o 4 referencias manuales proporcionadas por la interfaz.
        """
        if strip_idx < 0 or strip_idx >= len(self._strips_visual):
            logger.error(f"Índice de strip inválido: {strip_idx}")
            return None

        sv = self._strips_visual[strip_idx]
        if 'strip_orig' not in sv:
            logger.error(f"No hay imagen original guardada para tirilla {strip_idx}.")
            return None

        intensidades, posiciones, r = self._muestrear_y_posiciones(
            sv.get('strip_pos', sv['strip_orig']), tl, tr, bl, pbs_bot,
            img_muestreo=sv['strip_orig'], mask_sat=sv.get('mask_sat'))
        sv['alertas'] = dict(self._ultimas_alertas)
        sv['alertas']['recalibrada_manual'] = True
        sv['refs'] = (tl, tr, bl, pbs_bot)
        # Actualizar cache para que el mapa visual use las nuevas posiciones
        sv['posiciones_cache'] = posiciones
        sv['r_cache'] = r
        logger.info(
            f"Tirilla {strip_idx + 1} recalibrada manualmente: "
            f"TL=({tl[0]:.0f},{tl[1]:.0f}) "
            f"TR=({tr[0]:.0f},{tr[1]:.0f}) "
            f"BL=({bl[0]:.0f},{bl[1]:.0f})"
        )
        return intensidades

    # ------------------------------------------------------------------
    def obtener_alertas(self) -> List[Dict]:
        """
        Alertas de calidad por tirilla, para mostrarlas en el reporte:
          sin_referencias, discordantes (lista), saturadas (lista),
          recalibrada_manual.
        """
        return [dict(sv.get('alertas', {})) for sv in self._strips_visual]

    # ------------------------------------------------------------------
    def obtener_datos_mapa_visual(self) -> list:
        """
        Devuelve para cada tirilla:
          - 'strip_img': imagen de la tirilla (siempre, aunque falle auto-detección)
          - 'posiciones': dict {(col, fila): (px, py, proteina)} (vacío si auto falló)
          - 'auto_failed': True si la detección automática no encontró referencias
          - 'offset': (x0, y0, w, h) del strip en la imagen original

        Esto permite que la interfaz muestre la imagen incluso cuando la
        detección automática falló, para que el usuario pueda clickear
        los Reference Spots manualmente.
        """
        datos = []
        for sv in self._strips_visual:
            item = {
                'strip_img': sv.get('strip_img'),
                'offset': sv.get('offset'),
                'auto_failed': sv.get('refs') is None,
                'posiciones': {},
                'refs_puntos': sv.get('refs'),
            }
            if sv.get('refs') is not None:
                refs = sv['refs']
                tl, tr, bl = refs[0], refs[1], refs[2]
                pbs_bot = refs[3] if len(refs) > 3 else None

                strip_orig = sv.get('strip_orig')
                if strip_orig is not None:
                    # Usar posiciones cacheadas si están disponibles (evita re-ejecutar
                    # el pipeline completo — ya se calculó en analizar_imagen).
                    # Solo recalculamos si fue una recalibración manual posterior.
                    posiciones_cache = sv.get('posiciones_cache')
                    r_cache = sv.get('r_cache')

                    if posiciones_cache is not None and r_cache is not None:
                        posiciones = posiciones_cache
                        radio = r_cache
                        # Imagen invertida en escala 0-255 para _calcular_calidad
                        # (debe ser la MISMA escala que usa el pipeline interno)
                        if strip_orig.dtype == np.uint16:
                            # Convertir 16-bit a 8-bit antes de invertir
                            gris_8 = (strip_orig // 256).astype(np.uint8)
                            img_inv = 255 - gris_8
                        elif strip_orig.dtype == np.uint8:
                            img_inv = 255 - strip_orig
                        else:
                            # Float u otro: normalizar a 0-255 y invertir
                            norm = cv2.normalize(strip_orig, None, 0, 255,
                                                  cv2.NORM_MINMAX).astype(np.uint8)
                            img_inv = 255 - norm
                    else:
                        # Sin cache (ej. recalibración manual): ejecutar pipeline completo
                        posiciones, img_inv, radio = self._calcular_posiciones_finales(
                            strip_orig, tl, tr, bl, pbs_bot)

                    # Calcular score de calidad y asignar proteínas
                    for (col, fila), (px, py) in posiciones.items():
                        calidad = self._calcular_calidad(img_inv, px, py, radio)
                        proteina = MAPA_PROTEINAS.get((col, fila), '?')
                        item['posiciones'][(col, fila)] = (
                            px, py, proteina, calidad,
                        )
            datos.append(item)
        return datos


if __name__ == "__main__":
    print("Motor de Apoptosis ARY009 cargado correctamente.")
    print(f"Proteínas en el mapa: {len(PROTEINAS_UNICAS)}")
