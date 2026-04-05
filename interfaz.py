import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import traceback
import os

# Importamos las clases del archivo procesador que renombramos a motor_apoptosis.py
try:
    from motor_apoptosis import ProcesadorTirillas, AnalizadorApoptosis
except ImportError:
    messagebox.showerror(
        "Error Crítico", 
        "No se pudo importar 'motor_apoptosis.py'. Asegúrese de que el archivo esté en la misma carpeta que este ejecutable."
    )

# Configuración visual moderna con CustomTkinter
ctk.set_appearance_mode("dark")  # Modo oscuro
ctk.set_default_color_theme("blue")  # Tema azul (botones, sliders, etc.)

class AplicacionApoptosis(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana principal
        self.title("Analizador de Apoptosis - ITM")
        self.geometry("800x650")
        self.minsize(700, 600)
        
        # Variables para almacenar rutas de archivo en memoria
        self.ruta_control = None
        self.ruta_tratamiento = None

        # ----------------- TÍTULO -----------------
        self.lbl_titulo = ctk.CTkLabel(
            self, 
            text="Analizador de Apoptosis - ITM", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.lbl_titulo.pack(pady=(20, 20))

        # ----------------- PANEL SUPERIOR (Botones de Carga) -----------------
        self.frame_cargas = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_cargas.pack(fill="x", padx=40, pady=10)
        
        # Grid layout para centrar proporciones
        self.frame_cargas.grid_columnconfigure(0, weight=1)
        self.frame_cargas.grid_columnconfigure(1, weight=1)

        # -- Sección Imagen Control --
        self.frame_control = ctk.CTkFrame(self.frame_cargas)
        self.frame_control.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.btn_cargar_control = ctk.CTkButton(
            self.frame_control, 
            text="Cargar Imagen Control", 
            command=self.cargar_control
        )
        self.btn_cargar_control.pack(pady=(20, 10))
        
        self.lbl_ruta_control = ctk.CTkLabel(
            self.frame_control, 
            text="Ningún archivo seleccionado...", 
            text_color="gray",
            wraplength=300
        )
        self.lbl_ruta_control.pack(pady=(0, 20), padx=10)

        # -- Sección Imagen Tratamiento --
        self.frame_tratamiento = ctk.CTkFrame(self.frame_cargas)
        self.frame_tratamiento.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.btn_cargar_tratamiento = ctk.CTkButton(
            self.frame_tratamiento, 
            text="Cargar Imagen Tratamiento", 
            command=self.cargar_tratamiento
        )
        self.btn_cargar_tratamiento.pack(pady=(20, 10))
        
        self.lbl_ruta_tratamiento = ctk.CTkLabel(
            self.frame_tratamiento, 
            text="Ningún archivo seleccionado...", 
            text_color="gray",
            wraplength=300
        )
        self.lbl_ruta_tratamiento.pack(pady=(0, 20), padx=10)

        # ----------------- BOTÓN DE ANÁLISIS -----------------
        # Usamos un verde llamativo para la acción principal
        self.btn_analizar = ctk.CTkButton(
            self, 
            text="Analizar Membranas", 
            fg_color="#28a745", 
            hover_color="#218838",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            command=self.analizar_membranas
        )
        self.btn_analizar.pack(pady=20)

        # ----------------- CAJA DE RESULTADOS -----------------
        self.lbl_r = ctk.CTkLabel(self, text="Resultados del Análisis:")
        self.lbl_r.pack(anchor="w", padx=40)

        self.textbox_resultados = ctk.CTkTextbox(self, width=700, height=250, font=ctk.CTkFont(family="Consolas"))
        self.textbox_resultados.pack(padx=40, pady=(5, 20), fill="both", expand=True)

    # ----------------- MÉTODOS DE LA CLASE -----------------
    
    def cargar_control(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen de Control",
            filetypes=[
                ("Imágenes soportadas", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.scn *.svs *.ndpi"),
                ("Leica WSI", "*.scn"),
                ("Imágenes estándar", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        if ruta:
            self.ruta_control = ruta
            # Mostrar solo el nombre del archivo para no saturar la UI
            self.lbl_ruta_control.configure(text=os.path.basename(ruta), text_color=("black", "white"))

    def cargar_tratamiento(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen de Tratamiento",
            filetypes=[
                ("Imágenes soportadas", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.scn *.svs *.ndpi"),
                ("Leica WSI", "*.scn"),
                ("Imágenes estándar", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        if ruta:
            self.ruta_tratamiento = ruta
            self.lbl_ruta_tratamiento.configure(text=os.path.basename(ruta), text_color=("black", "white"))

    def analizar_membranas(self):
        # 1. Validaciones iniciales
        if not self.ruta_control or not self.ruta_tratamiento:
            messagebox.showwarning("Faltan Archivos", "Por favor cargue ambas imágenes (Control y Tratamiento) antes de analizar.")
            return
            
        # Limpiar la caja de texto antes de una nueva corrida
        self.textbox_resultados.delete("1.0", ctk.END)
        self.textbox_resultados.insert(ctk.END, "Iniciando análisis de densitometría...\n\n")
        self.update() # Refrescar la GUI para que el usuario vea el mensaje

        # 2. Bloque Try-Except global para evitar que la aplicación 'crashee'
        try:
            # Diccionario Teórico (Ejemplo estático para este proyecto académico)
            # A futuro, esto puede venir de un archivo Excel .csv seleccionable
            cuadricula_teorica = {
                'Reference_1': (10,  10, 15, 15),
                'Reference_2': (30,  10, 15, 15),
                'Bax':         (100, 200, 20, 20),
                'Caspase-3':   (150, 200, 20, 20),
                'Bcl-2':       (10,  10, 20, 20),
                'Bad':         (50,  50, 20, 20),
            }
            REFERENCIAS = ['Reference_1', 'Reference_2']

            # Instanciación de Clases
            self.textbox_resultados.insert(ctk.END, "-> Cargando imágenes en Motor OpenCV...\n")
            procesador_control = ProcesadorTirillas(self.ruta_control)
            procesador_experimento = ProcesadorTirillas(self.ruta_tratamiento)

            # Preprocesamiento (Alineación con Canny/Perspectiva y Bitwise Not)
            self.textbox_resultados.insert(ctk.END, "-> Alineando perspectivas e invirtiendo matrices de grises...\n")
            img_control = procesador_control.preprocesar_imagen()
            img_experimento = procesador_experimento.preprocesar_imagen()

            # Extracción de ROI's
            self.textbox_resultados.insert(ctk.END, "-> Extrayendo intensidades luminosas...\n")
            datos_ctrl = procesador_control.extraer_intensidad_puntos(img_control, cuadricula_teorica)
            datos_exp = procesador_experimento.extraer_intensidad_puntos(img_experimento, cuadricula_teorica)

            # Cálculos de Expresión y Fold Change
            self.textbox_resultados.insert(ctk.END, "-> Calculando Fold Change y Resumen Clínico...\n\n")
            analizador = AnalizadorApoptosis(
                pro_apoptoticas=['Bax', 'Caspase-3', 'Bad'],
                anti_apoptoticas=['Bcl-2']
            )

            ctrl_norm = analizador.normalizar_por_referencia(datos_ctrl, REFERENCIAS)
            exp_norm = analizador.normalizar_por_referencia(datos_exp, REFERENCIAS)

            reporte_cientifico = analizador.calcular_fold_change(ctrl_norm, exp_norm)

            # 3. Presentación de los resultados estructurados
            texto_json = json.dumps(reporte_cientifico, indent=4, ensure_ascii=False)
            
            self.textbox_resultados.insert(ctk.END, "================ REPORTE CIENTÍFICO ================\n")
            self.textbox_resultados.insert(ctk.END, texto_json)
            self.textbox_resultados.insert(ctk.END, "\n====================================================")
            
        except Exception as e:
            # Si ocurre cualquier error geométrico, lectura, dict key error, lo capturamos aquí
            error_mensaje = traceback.format_exc()
            self.textbox_resultados.insert(ctk.END, "\n[X] Ocurrió un error inesperado durante el análisis:\n\n")
            self.textbox_resultados.insert(ctk.END, error_mensaje)
            
            # También mostramos una alerta visual invasiva para asegurarnos que el usuario lo note
            messagebox.showerror("Error en Análisis", f"Revise el Textbox para leer la traza completa.\nError: {str(e)}")


# Ejecución de la ventana principal
if __name__ == "__main__":
    app = AplicacionApoptosis()
    app.mainloop()
