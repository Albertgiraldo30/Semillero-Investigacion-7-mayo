import cv2
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(name)s]: %(message)s')
logger = logging.getLogger('AnalizadorApoptosis')

# Umbral mínimo de intensidad para considerar que una proteína está realmente presente.
# Intensidades por debajo de este valor se tratan como ruido de fondo del array.
EPSILON_INTENSIDAD = 1.0


class ProcesadorTirillas:
    """
    Clase para procesar imágenes de membranas del Proteome Profiler Human Apoptosis Array.
    """

    def __init__(self, ruta_imagen, umbral_canny_min=50, umbral_canny_max=150, epsilon_poly=0.02):
        """
        Inicializa el ProcesadorTirillas con la ruta y los parámetros de procesamiento.

        Args:
            ruta_imagen (str): La ruta al archivo de la imagen de la membrana.
            umbral_canny_min (int): Umbral mínimo para el detector de bordes Canny.
            umbral_canny_max (int): Umbral máximo para el detector de bordes Canny.
            epsilon_poly (float): Tolerancia para la aproximación poligonal (fracción del perímetro).
        """
        self.ruta_imagen = ruta_imagen
        self.umbral_canny_min = umbral_canny_min
        self.umbral_canny_max = umbral_canny_max
        self.epsilon_poly = epsilon_poly

    def alinear_tirilla(self, imagen):
        """
        Detecta el contorno rectangular de la membrana y aplica una transformación
        de perspectiva para enderezarla y recortarla sin áreas vacías en las esquinas.

        A diferencia de cv2.boundingRect() (que produce un recorte ortogonal con
        esquinas vacías si la membrana está rotada), getPerspectiveTransform() mapea
        las 4 esquinas reales al rectángulo destino, eliminando la rotación.

        Args:
            imagen (numpy.ndarray): La imagen original BGR.

        Returns:
            numpy.ndarray: La imagen recortada y enderezada de la membrana, a color.
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

        # --- MEJORA: transformación de perspectiva en lugar de boundingRect ---
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
    def _ordenar_puntos(puntos):
        """
        Ordena 4 puntos: [superior-izq, superior-der, inferior-der, inferior-izq].

        La suma (x+y) identifica las esquinas diagonal-mente opuestas;
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

    def preprocesar_imagen(self):
        """
        Lee la imagen, alinea la membrana, convierte a escala de grises e invierte.

        La inversión es necesaria porque en la membrana original los puntos de proteína
        aparecen oscuros sobre fondo claro. Al invertir, el fondo queda en 0 (negro)
        y los puntos quedan en valores altos (blancos), de modo que la intensidad
        promedio de un ROI es directamente proporcional a la concentración de proteína.

        Returns:
            numpy.ndarray: La imagen en escala de grises invertida.
        """
        imagen = cv2.imread(self.ruta_imagen)
        if imagen is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en: {self.ruta_imagen}")

        imagen_alineada = self.alinear_tirilla(imagen)
        imagen_gris = cv2.cvtColor(imagen_alineada, cv2.COLOR_BGR2GRAY)
        imagen_invertida = cv2.bitwise_not(imagen_gris)
        return imagen_invertida

    def extraer_intensidad_puntos(self, imagen_procesada, rois):
        """
        Extrae la intensidad promedio (0–255) de cada región de interés (ROI).

        Args:
            imagen_procesada (numpy.ndarray): La imagen en escala de grises invertida.
            rois (dict): Diccionario {nombre: (x, y, ancho, alto)}.

        Returns:
            dict: {nombre: intensidad_promedio} donde None indica ROI inválido.
                  Los ROIs inválidos se devuelven como None (no como 0.0) para que
                  el analizador pueda distinguir "ausencia real" de "dato no medible".
        """
        alto_imagen, ancho_imagen = imagen_procesada.shape[:2]
        resultados = {}

        for nombre, coordenadas in rois.items():
            x, y, ancho, alto = coordenadas

            # --- MEJORA: validar límites antes de recortar ---
            # NumPy recorta silenciosamente los ROIs fuera de límites, generando
            # mediciones incorrectas sin ningún error visible. Detectamos esto explícitamente.
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


class AnalizadorApoptosis:
    """
    Clase para analizar los resultados del Proteome Profiler Human Apoptosis Array
    usando Densidad Óptica Relativa normalizada (Fold Change).
    """

    def __init__(self, pro_apoptoticas=None, anti_apoptoticas=None):
        """
        Inicializa el analizador con las listas de proteínas por categoría biológica.

        Args:
            pro_apoptoticas (list): Nombres de proteínas pro-apoptóticas (ej. Bax, Bad).
            anti_apoptoticas (list): Nombres de proteínas anti-apoptóticas (ej. Bcl-2, XIAP).
        """
        self.pro_apoptoticas = pro_apoptoticas or ['Bax', 'Bad', 'Cytochrome c', 'Caspase-3']
        self.anti_apoptoticas = anti_apoptoticas or ['Bcl-2', 'Bcl-xL', 'Survivin', 'XIAP']

    def normalizar_por_referencia(self, intensidades, nombres_referencia):
        """
        Normaliza las intensidades dividiéndolas por el promedio de los puntos
        de referencia positivos del array (Reference Spots).

        Los arrays Proteome Profiler incluyen duplicados de puntos de referencia
        positivos (manchas de anticuerpo conocido) en cada membrana. Normalizar
        por su promedio elimina las variaciones técnicas de exposición entre
        membranas (distintos tiempos de revelado, cantidad de muestra cargada, etc.),
        haciendo que el fold change refleje diferencias biológicas reales.

        Args:
            intensidades (dict): Intensidades crudas {nombre: valor}.
            nombres_referencia (list): Lista de nombres de los puntos de referencia
                                       positivos en el diccionario de intensidades.

        Returns:
            dict: Intensidades normalizadas. Si no hay referencias válidas, retorna
                  las intensidades originales con una advertencia.
        """
        valores_ref = [
            intensidades[n] for n in nombres_referencia
            if n in intensidades and intensidades[n] is not None and intensidades[n] > EPSILON_INTENSIDAD
        ]

        if not valores_ref:
            logger.warning(
                "No se encontraron puntos de referencia válidos para normalizar. "
                "Se usarán las intensidades crudas (los fold changes pueden no ser comparables entre membranas)."
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

    def calcular_fold_change(self, intensidades_control, intensidades_tratamiento):
        """
        Calcula el Fold Change (Tratamiento / Control) para cada proteína presente.

        Reglas de cálculo:
        - Control > ε y Tratamiento > ε  → Fold = Tratamiento / Control (caso normal)
        - Control < ε y Tratamiento > ε  → Proteína de novo; se reporta como float('inf')
        - Control > ε y Tratamiento < ε  → Proteína silenciada; se reporta como 0.0
        - Ambas < ε                       → Proteína ausente; se EXCLUYE del reporte
        - Cualquier valor None            → Medición inválida; se EXCLUYE del reporte

        La exclusión de proteínas ausentes en ambas condiciones es científicamente
        necesaria: reportarlas con fold=1.0 implicaría falsamente que el tratamiento
        no las afectó, cuando en realidad no estaban presentes en el experimento.

        Args:
            intensidades_control (dict): Intensidades normalizadas del control.
            intensidades_tratamiento (dict): Intensidades normalizadas del tratamiento.

        Returns:
            dict: Reporte estructurado {categoría: {proteína: fold_change}},
                  más un campo 'Resumen' con métricas derivadas del experimento.
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

            # --- MEJORA: excluir mediciones inválidas (None) del reporte ---
            if val_ctrl is None or val_trat is None:
                logger.warning(f"Proteína '{proteina}' tiene medición inválida (None). Excluida del reporte.")
                continue

            # --- MEJORA: comparación con epsilon en lugar de == 0.0 exacto ---
            # La igualdad exacta con flotantes es frágil: 0.0001 < EPSILON no es 0.0.
            ctrl_ausente = val_ctrl < EPSILON_INTENSIDAD
            trat_ausente = val_trat < EPSILON_INTENSIDAD

            # --- MEJORA: excluir proteínas ausentes en ambas condiciones ---
            if ctrl_ausente and trat_ausente:
                logger.info(f"Proteína '{proteina}' ausente en ambas condiciones. Excluida del reporte.")
                continue

            if ctrl_ausente:
                # Expresión de novo inducida por el tratamiento
                fold_change = float('inf')
                logger.info(f"Proteína '{proteina}': expresión de novo (control=0, trat={val_trat:.4f}).")
            elif trat_ausente:
                # Silenciamiento completo por el tratamiento
                fold_change = 0.0
                logger.info(f"Proteína '{proteina}': silenciada completamente (trat≈0).")
            else:
                fold_change = round(val_trat / val_ctrl, 4)

            # Clasificar en la categoría biológica correspondiente
            if proteina in self.pro_apoptoticas:
                reporte['Pro-apoptóticas'][proteina] = fold_change
            elif proteina in self.anti_apoptoticas:
                reporte['Anti-apoptóticas'][proteina] = fold_change
            else:
                reporte['No Clasificadas (Otras)'][proteina] = fold_change

        # --- MEJORA: agregar métricas de resumen al reporte ---
        reporte['Resumen'] = self._calcular_resumen(reporte)
        return reporte

    def _calcular_resumen(self, reporte):
        """
        Calcula métricas derivadas del reporte de fold change para facilitar
        la interpretación biológica del experimento.

        Args:
            reporte (dict): Reporte con categorías y sus fold changes.

        Returns:
            dict: Métricas de resumen incluyendo proteínas más reguladas y el
                  ratio pro/anti-apoptótico del tratamiento.
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

        # El ratio pro/anti-apoptótico resume si el tratamiento empuja la célula
        # hacia apoptosis (ratio > 1) o hacia supervivencia (ratio < 1).
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
                'Tendencia pro-apoptótica' if ratio_pro_anti and ratio_pro_anti > 1.0
                else 'Tendencia anti-apoptótica' if ratio_pro_anti and ratio_pro_anti < 1.0
                else 'No determinada'
            )
        }


# ==========================================
# Ejemplo de uso
# ==========================================
if __name__ == "__main__":
    # procesador_control = ProcesadorTirillas("membrana_sana.jpg")
    # procesador_experimento = ProcesadorTirillas("membrana_tratada.jpg")
    #
    # try:
    #     img_control = procesador_control.preprocesar_imagen()
    #     img_experimento = procesador_experimento.preprocesar_imagen()
    #
    #     # Las coordenadas del manual del array (incluir Reference Spots)
    #     cuadricula = {
    #         'Reference_1': (10,  10, 15, 15),
    #         'Reference_2': (30,  10, 15, 15),
    #         'Bax':         (100, 200, 20, 20),
    #         'Caspase-3':   (150, 200, 20, 20),
    #         'Bcl-2':       (10,  10, 20, 20),
    #         'Bad':         (50,  50, 20, 20),
    #     }
    #     REFERENCIAS = ['Reference_1', 'Reference_2']
    #
    #     datos_ctrl = procesador_control.extraer_intensidad_puntos(img_control, cuadricula)
    #     datos_exp  = procesador_experimento.extraer_intensidad_puntos(img_experimento, cuadricula)
    #
    #     analizador = AnalizadorApoptosis(
    #         pro_apoptoticas=['Bax', 'Caspase-3', 'Bad'],
    #         anti_apoptoticas=['Bcl-2']
    #     )
    #
    #     # Normalizar por Reference Spots antes de calcular fold change
    #     ctrl_norm = analizador.normalizar_por_referencia(datos_ctrl, REFERENCIAS)
    #     exp_norm  = analizador.normalizar_por_referencia(datos_exp,  REFERENCIAS)
    #
    #     reporte = analizador.calcular_fold_change(ctrl_norm, exp_norm)
    #
    #     import json
    #     print("===== REPORTE FOLD CHANGE =====")
    #     print(json.dumps(reporte, indent=4, ensure_ascii=False))
    #
    # except Exception as e:
    #     logger.exception("Error procesando el array.")
    pass