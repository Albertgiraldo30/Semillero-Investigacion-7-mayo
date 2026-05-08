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
    COORDS_REFERENCIA, COORDS_PBS, PARES_DUPLICADOS,
)

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
            
        # Intensidad: medimos en la imagen original invertida para tener valores 0-255 (255=máx intensidad)
        invertida = cv2.bitwise_not(gris)
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
    # En el ARY009 la columna E (izquierda) está separada por un espacio mayor
    # respecto al bloque central (D, C, B) y A.
    _T_COLS = {'A': 0.0, 'B': 0.22, 'C': 0.44, 'D': 0.66, 'E': 1.0}
    # Centroide de los pares de referencia en coordenadas de fila (1-24)
    _FILA_REF_TOP = 1.5    # centro de rows 1 y 2
    _FILA_REF_BOT = 23.5   # centro de rows 23 y 24

    def __init__(self, radio_muestreo: int = 8):
        self.radio = radio_muestreo
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

        Estrategia robusta para fondos blancos Y grises:
        1. Black-Hat para aislar manchas oscuras (elimina el nivel de fondo)
        2. Proyección del mapa de manchas → solo los spots contribuyen
        3. Localiza límites de contenido y, si hay N valles claros, los usa;
           en caso contrario divide el rango de contenido uniformemente en N.
        """
        h, w = imagen_gris.shape

        # 1. Transformación morfológica (BlackHat para fondo claro, TopHat para fondo oscuro)
        max_val = np.iinfo(imagen_gris.dtype).max if imagen_gris.dtype.kind == 'u' else 255
        fondo_claro = np.mean(imagen_gris) > (max_val * 0.35)
        
        tam_k = 51
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_k, tam_k))
        if fondo_claro:
            bhat = cv2.morphologyEx(imagen_gris, cv2.MORPH_BLACKHAT, kernel_bh)
        else:
            bhat = cv2.morphologyEx(imagen_gris, cv2.MORPH_TOPHAT, kernel_bh)

        # 2. Binarizar el resultado del Black-Hat
        _, bin_bhat = cv2.threshold(bhat, 8, 255, cv2.THRESH_BINARY)

        # 3. Proyección horizontal de los spots binarios
        proj_bhat = bin_bhat.astype(np.float64).sum(axis=0)

        # 4. Suavizado moderado para rellenar huecos entre columnas de un strip
        smooth = max(5, w // (n * 6))
        proj_s = np.convolve(proj_bhat, np.ones(smooth) / smooth, mode='same')

        if proj_s.max() < 1:
            logger.warning("Black-Hat no detectó manchas; fallback a inversión.")
            inv = cv2.bitwise_not(imagen_gris)
            proj_s = np.convolve(inv.astype(np.float64).sum(axis=0),
                                 np.ones(15) / 15, mode='same')

        proj_norm = proj_s - proj_s.min()
        if proj_norm.max() < 1:
            logger.warning("Proyección plana; no se pueden separar strips.")
            return []

        # 5. Límites globales de contenido
        thresh_global = proj_norm.max() * 0.05
        xs_content = np.where(proj_norm > thresh_global)[0]
        x_min, x_max = int(xs_content[0]), int(xs_content[-1])
        span = max(1, x_max - x_min)

        # 6. Intentar detectar N segmentos con valleys entre strips
        #    (funciona cuando los gaps son visibles en la proyección)
        thresh_seg = proj_norm.max() * 0.20
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

        # Fusionar segmentos muy cercanos (huecos dentro de un mismo strip)
        gap_max = span // (n * 2)
        merged: List[List[int]] = []
        for s in sorted(segs, key=lambda x: x[0]):
            if merged and s[0] - merged[-1][1] < gap_max:
                merged[-1][1] = s[1]
            else:
                merged.append([s[0], s[1]])

        ancho_min = span // (n * 3)
        merged = [s for s in merged if s[1] - s[0] > ancho_min]

        if len(merged) == n:
            final_segs = [(s[0], s[1]) for s in merged]
            logger.info("Separación por segmentos detectados.")
        else:
            # Fallback: división uniforme del rango de contenido
            strip_w = span / n
            final_segs = [
                (int(x_min + i * strip_w), int(x_min + (i + 1) * strip_w))
                for i in range(n)
            ]
            logger.info(f"Fallback a división uniforme ({len(merged)} seg → {n} strips).")

        # 7. Para cada segmento, recortar con márgenes y ajustar límites verticales
        mg_x = max(8, span // (n * 8))
        strips = []
        for x0, x1 in final_segs:
            x0 = max(0, x0 - mg_x)
            x1 = min(w, x1 + mg_x)

            col_bhat = bin_bhat[:, x0:x1]
            proj_y = col_bhat.astype(np.float64).sum(axis=1)
            rows = np.where(proj_y > 0)[0]
            if rows.size == 0:
                y0, y1 = 0, h
            else:
                mg_y = 12
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

        Intenta primero la estrategia por PARES (más confiable). Si falla,
        usa la estrategia legacy (mayor blob por cuadrante).
        """
        # Intento principal: pares verticales en las esquinas
        resultado = self._referencias_por_pares(strip_gris)
        if resultado is not None:
            return resultado

        # Fallback: método antiguo (mayor blob en cada cuadrante)
        logger.info("Detección por pares falló. Probando método legacy.")
        return self._referencias_legacy(strip_gris)

    # ------------------------------------------------------------------
    def _detectar_spots_para_referencias(
        self, strip_gris: np.ndarray, estricto: bool = True,
    ) -> List[Dict]:
        """Detecta candidatos a Reference Spot con filtros configurables."""
        h, w = strip_gris.shape

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(strip_gris)

        tam_k = max(25, min(w // 3, 51))
        if tam_k % 2 == 0:
            tam_k += 1
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

        area_min = 8 if estricto else 4
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
        con la geometría general de la membrana ARY009.

        Si estricto=True: aspect ratio entre 3.0 y 9.0, áreas similares.
        Si estricto=False: solo tamaños mínimos (mucho más permisivo).
        """
        sep_x = abs(tr_c['cx'] - tl_c['cx'])
        sep_y_left = abs(bp['cy'] - tl_c['cy'])
        sep_y_right = abs(bp['cy'] - tr_c['cy'])
        sep_y = max(sep_y_left, sep_y_right)

        if sep_x < 15 or sep_y < 40:
            return False

        if not estricto:
            # Modo permisivo: solo tamaños mínimos
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
        Estrategia legacy: el blob más grande en cada cuadrante (relativo a las manchas).
        Probamos validación estricta primero; si falla, aceptamos cualquier
        L-shape no degenerado.
        """
        h, w = strip_gris.shape
        spots = self._detectar_spots_para_referencias(strip_gris, estricto=False)
        if len(spots) < 3:
            return None

        min_x = min(s['cx'] for s in spots)
        max_x = max(s['cx'] for s in spots)
        min_y = min(s['cy'] for s in spots)
        max_y = max(s['cy'] for s in spots)
        strip_w = max(max_x - min_x, 10.0)
        strip_h = max(max_y - min_y, 10.0)

        def mayor_en(x0, x1, y0, y1):
            cands = [b for b in spots if min_x + strip_w * x0 <= b['cx'] <= min_x + strip_w * x1 
                     and min_y + strip_h * y0 <= b['cy'] <= min_y + strip_h * y1]
            return max(cands, key=lambda b: b['area']) if cands else None

        candidatos = []
        for f in [0.35, 0.45, 0.55, 0.65]:
            tl = mayor_en(0, f, 0, f)
            tr = mayor_en(1 - f, 1.0, 0, f)
            bl = mayor_en(0, f, 1 - f, 1.0)
            br = mayor_en(1 - f, 1.0, 1 - f, 1.0)

            for bot_candidate in (bl, br):
                if not (tl and tr and bot_candidate):
                    continue
                candidatos.append((tl, tr, bot_candidate))

        dy = (strip_h / 23.0) * 0.5

        for tl, tr, bp in candidatos:
            if self._validar_l_shape(tl, tr, bp, int(strip_w), int(strip_h), estricto=True):
                tl_c = {'cx': tl['cx'], 'cy': tl['cy'] + dy, 'area': tl['area']}
                tr_c = {'cx': tr['cx'], 'cy': tr['cy'] + dy, 'area': tr['area']}
                bp_c = {'cx': bp['cx'], 'cy': bp['cy'] - dy, 'area': bp['area']}
                return self._asignar_anchors(tl_c, tr_c, bp_c)

        for tl, tr, bp in candidatos:
            if self._validar_l_shape(tl, tr, bp, int(strip_w), int(strip_h), estricto=False):
                logger.info("Legacy: usando candidato con validación permisiva.")
                tl_c = {'cx': tl['cx'], 'cy': tl['cy'] + dy, 'area': tl['area']}
                tr_c = {'cx': tr['cx'], 'cy': tr['cy'] + dy, 'area': tr['area']}
                bp_c = {'cx': bp['cx'], 'cy': bp['cy'] - dy, 'area': bp['area']}
                return self._asignar_anchors(tl_c, tr_c, bp_c)

        logger.warning("Legacy: no se encontró un L-shape válido.")
        return None

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

        logger.info(
            f"Referencias: TL=({tl[0]:.0f},{tl[1]:.0f})  "
            f"TR=({tr[0]:.0f},{tr[1]:.0f})  "
            f"BL=({bl[0]:.0f},{bl[1]:.0f})"
        )
        return (tl, tr, bl)

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

        py = tl[1] + t_row * (bl[1] - tl[1]) + t_col * (tr[1] - tl[1])

        x_A = tl[0] + t_row * (bl[0] - tl[0])
        
        if pbs_bot is not None:
            # pbs_bot es D23/D24 (t_col = 0.66, t_row = 1.0)
            x_D_top = tl[0] + 0.66 * (tr[0] - tl[0])
            x_D_bot = pbs_bot[0]
            x_D = x_D_top + t_row * (x_D_bot - x_D_top)
            
            w_total = (x_D - x_A) / 0.66
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

            # Límites físicos: no más de la mitad del paso entre filas/cols
            # 0.11 = mitad de la separación de columnas (0.22 en t_col)
            max_shift_x = max(2, int(abs(tr[0] - tl[0]) * 0.11))
            row_spacing = abs(bl[1] - tl[1]) / 23.0 if abs(bl[1] - tl[1]) > 0 else ry * 2
            max_shift_y = max(2, int(row_spacing * 0.45))

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
        # Normalizar a 8-bit para que los umbrales sean estables
        if strip_gris.dtype != np.uint8:
            gris_8 = cv2.normalize(strip_gris, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
        else:
            gris_8 = strip_gris.copy()

        # Suavizado ligero para reducir ruido fino
        gris_8 = cv2.GaussianBlur(gris_8, (3, 3), 0)

        # Multi-escala: 3 kernels para capturar spots de distintos tamaños
        mascara_combinada = None
        for k_size in (11, 17, 23):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            bhat = cv2.morphologyEx(gris_8, cv2.MORPH_BLACKHAT, kernel)
            if bhat.max() < 4:
                continue
            # Umbral más sensible (12% del máximo) para capturar tenues
            umbral = max(4.0, float(bhat.max()) * 0.12)
            _, m = cv2.threshold(bhat, umbral, 255, cv2.THRESH_BINARY)
            if mascara_combinada is None:
                mascara_combinada = m
            else:
                mascara_combinada = cv2.bitwise_or(mascara_combinada, m)

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
          - Requiere al menos 8 correspondencias válidas
        """
        if len(spots) < 8:
            return pos_dict

        spots_xy = np.array([[s[0], s[1]] for s in spots], dtype=np.float32)
        max_dist = max(row_spacing, col_spacing) * 0.40
        max_dist_sq = max_dist * max_dist

        pos = dict(pos_dict)

        for _ in range(max_iter):
            src_list, dst_list = [], []
            for (col, fila), (px, py) in pos.items():
                dx = spots_xy[:, 0] - px
                dy = spots_xy[:, 1] - py
                d_sq = dx * dx + dy * dy
                min_idx = int(np.argmin(d_sq))
                if d_sq[min_idx] < max_dist_sq:
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

        # Anillo de fondo: 3× el radio, excluyendo el spot
        bg_r = r * 3
        bx0, bx1 = max(0, px - bg_r), min(w, px + bg_r)
        by0, by1 = max(0, py - bg_r), min(h, py + bg_r)
        bg_region = inv_img[by0:by1, bx0:bx1]
        if bg_region.size == 0:
            return 0.5
        # Estimar fondo con percentil bajo (lo más oscuro del entorno = fondo)
        bg_signal = float(np.percentile(bg_region, 25))

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

        # Radios de búsqueda para el Safe Snapping (mean-shift local)
        rx_safe = max(1, int(abs(tr[0] - tl[0]) * 0.08))
        ry_safe = max(1, r)

        # Detección global de spots (multi-escala Black-Hat)
        spots_detectados = self._detectar_todos_los_spots(strip_gris)
        row_spacing = abs(bl[1] - tl[1]) / 22.0 if abs(bl[1] - tl[1]) > 0 else r * 2
        col_spacing = abs(tr[0] - tl[0]) / 4.0 if abs(tr[0] - tl[0]) > 0 else r * 2
        max_snap_x = col_spacing * 0.40
        max_snap_y = row_spacing * 0.45

        def _snap_a_spot_cercano(px: int, py: int) -> Tuple[int, int]:
            """Snap a la mancha detectada más cercana dentro de los límites."""
            if not spots_detectados:
                return px, py
            mejor_spot = None
            mejor_dist = float('inf')
            for sx, sy, _sa in spots_detectados:
                dx = abs(sx - px)
                dy = abs(sy - py)
                if dx <= max_snap_x and dy <= max_snap_y:
                    dist = dx * dx + dy * dy
                    if dist < mejor_dist:
                        mejor_dist = dist
                        mejor_spot = (sx, sy)
            if mejor_spot is not None:
                return int(round(mejor_spot[0])), int(round(mejor_spot[1]))
            return px, py

        # ── PASADA 1: posición geométrica + mean-shift + snap global ──
        posiciones: Dict[Tuple[str, int], Tuple[int, int]] = {}
        for (col, fila), proteina in MAPA_PROTEINAS.items():
            px, py = self._pos_pixel(col, fila, tl, tr, bl, pbs_bot,
                                       img_inv=inv, rx=rx_safe, ry=ry_safe)
            px, py = _snap_a_spot_cercano(px, py)
            posiciones[(col, fila)] = (px, py)

        # ── PASADA 2: consistencia por filas (Y) y columnas (X) ──
        tol_y = max(3, int(row_spacing * 0.30))
        tol_x = max(3, int(col_spacing * 0.25))
        posiciones = self._aplicar_consistencia_filas(posiciones, tol_y)
        posiciones = self._aplicar_consistencia_columnas(posiciones, tol_x)

        # ── PASADA 2.5: ICP — alineación afín a los spots reales ──
        posiciones = self._icp_refinar(
            posiciones, spots_detectados, row_spacing, col_spacing,
        )

        # ── PASADA 3: re-snap final desde posición refinada ──
        for k, (px, py) in list(posiciones.items()):
            posiciones[k] = _snap_a_spot_cercano(px, py)

        return posiciones, inv, r

    # ------------------------------------------------------------------
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

        return {p: round(float(np.mean(v)), 2) for p, v in acum.items()}

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

        debug_img = None
        if ruta_debug:
            debug_img = cv2.cvtColor(gris_8, cv2.COLOR_GRAY2BGR)

        resultados: List[Dict[str, float]] = []
        self._strips_visual = []
        for i, (strip_8, (ox, oy, ow, oh)) in enumerate(strips):
            # Recortar la versión original (16-bit si aplica) para esta tirilla.
            # Se guarda para permitir re-muestreo con referencias manuales.
            strip_orig = gris_orig[oy:oy + oh, ox:ox + ow]

            refs = self.encontrar_referencias(strip_8)
            if refs is None:
                logger.warning(f"Strip {i+1}: sin referencias. Devolviendo intensidades vacías.")
                resultados.append({})
                self._strips_visual.append({
                    'strip_img': strip_8.copy(),
                    'strip_orig': strip_orig.copy(),
                    'refs': None,
                    'offset': (ox, oy, ow, oh),
                })
                continue

            tl, tr, bl = refs

            intensidades = self.muestrear_strip(strip_orig, tl, tr, bl)
            resultados.append(intensidades)

            if debug_img is not None:
                self._debug_strip(debug_img, strip_8, tl, tr, bl, ox, oy, i + 1)

            self._strips_visual.append({
                'strip_img': strip_8.copy(),
                'strip_orig': strip_orig.copy(),
                'refs': (tl, tr, bl),
                'offset': (ox, oy, ow, oh),
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

        if ext in wsi_exts:
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
    def _debug_strip(self, img, strip, tl, tr, bl, ox, oy, num):
        r = self.radio
        h_s, w_s = strip.shape

        for px, py, lbl in [(tl[0], tl[1], 'TL'), (tr[0], tr[1], 'TR'), (bl[0], bl[1], 'BL')]:
            cv2.circle(img, (int(ox + px), int(oy + py)), r + 4, (0, 140, 255), 2)
            cv2.putText(img, lbl, (int(ox + px) + r + 2, int(oy + py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        for (col, fila), proteina in MAPA_PROTEINAS.items():
            if proteina in ('Reference', 'PBS'):
                continue
            px, py = self._pos_pixel(col, fila, tl, tr, bl)
            color = ((0, 0, 200) if proteina in PRO_APOPTOTICAS else
                     (200, 0, 0) if proteina in ANTI_APOPTOTICAS else (0, 180, 0))
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

        intensidades = self.muestrear_strip(sv['strip_orig'], tl, tr, bl, pbs_bot)
        sv['refs'] = (tl, tr, bl, pbs_bot)
        logger.info(
            f"Tirilla {strip_idx + 1} recalibrada manualmente: "
            f"TL=({tl[0]:.0f},{tl[1]:.0f}) "
            f"TR=({tr[0]:.0f},{tr[1]:.0f}) "
            f"BL=({bl[0]:.0f},{bl[1]:.0f})"
        )
        return intensidades

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
                    # Reutilizar el pipeline unificado de posicionamiento
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
