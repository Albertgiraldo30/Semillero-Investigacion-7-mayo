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
        self.update()

        # 2. Bloque Try-Except global para evitar que la aplicación 'crashee'
        try:
            # ── PASO 1: Instanciar procesadores ──
            self.textbox_resultados.insert(ctk.END, "-> Cargando imágenes en Motor OpenCV...\n")
            self.update()
            procesador_control = ProcesadorTirillas(self.ruta_control)
            procesador_experimento = ProcesadorTirillas(self.ruta_tratamiento)

            # ── PASO 2: Pipeline Auto-ROI + Normalización por Referencia ──
            self.textbox_resultados.insert(ctk.END, "-> Mejorando imagen (denoising + CLAHE)...\n")
            self.textbox_resultados.insert(ctk.END, "-> Detectando manchas automáticamente (Otsu + contornos)...\n")
            self.textbox_resultados.insert(ctk.END, "-> Identificando Reference Spots y normalizando...\n")
            self.update()

            # Directorio del proyecto para guardar imágenes debug
            dir_proyecto = os.path.dirname(self.ruta_control)
            debug_ctrl = os.path.join(dir_proyecto, "debug_control.png")
            debug_trat = os.path.join(dir_proyecto, "debug_tratamiento.png")

            # detectar_y_extraer() ahora retorna 6 valores:
            # (crudas, normalizadas, muestras, referencias, imagen, promedio_ref)
            crudas_ctrl, norm_ctrl, muestras_ctrl, refs_ctrl, _, prom_ref_ctrl = \
                procesador_control.detectar_y_extraer(ruta_debug=debug_ctrl)
            crudas_exp, norm_exp, muestras_exp, refs_exp, _, prom_ref_exp = \
                procesador_experimento.detectar_y_extraer(ruta_debug=debug_trat)

            # Mostrar estadísticas de detección y referencia
            self.textbox_resultados.insert(ctk.END, f"\n   [Control]     Spots muestra: {len(muestras_ctrl)}, "
                                                    f"Refs: {len(refs_ctrl)}, "
                                                    f"Prom.Ref: {prom_ref_ctrl:.2f}\n")
            self.textbox_resultados.insert(ctk.END, f"   [Tratamiento] Spots muestra: {len(muestras_exp)}, "
                                                    f"Refs: {len(refs_exp)}, "
                                                    f"Prom.Ref: {prom_ref_exp:.2f}\n\n")
            self.update()

            if not norm_ctrl or not norm_exp:
                self.textbox_resultados.insert(ctk.END, 
                    "[!] ADVERTENCIA: No se detectaron manchas suficientes en una o ambas imágenes.\n"
                    "    Verifique que las imágenes sean de membranas del Proteome Profiler.\n"
                    "    Se necesitan al menos 4 spots (3 referencias + 1 muestra).\n"
                )
                return

            # ── PASO 3: Calcular Fold Change con intensidades NORMALIZADAS ──
            self.textbox_resultados.insert(ctk.END, "-> Calculando Fold Change (Intensidades Relativas)...\n\n")
            self.update()

            analizador = AnalizadorApoptosis()
            reporte_cientifico = analizador.calcular_fold_change(norm_ctrl, norm_exp)

            # ── PASO 4: Presentación de resultados ──
            texto_json = json.dumps(reporte_cientifico, indent=4, ensure_ascii=False)
            
            self.textbox_resultados.insert(ctk.END, "================ REPORTE CIENTÍFICO ================\n")
            self.textbox_resultados.insert(ctk.END, texto_json)
            self.textbox_resultados.insert(ctk.END, "\n====================================================\n\n")
            
            # Tabla de detalle: cruda, normalizada y fold
            self.textbox_resultados.insert(ctk.END, "──── DETALLE DE SPOTS (Normalizados por Referencia) ────\n")
            self.textbox_resultados.insert(ctk.END, f"{'Spot':<10} {'Cruda Ctrl':>11} {'Cruda Trat':>11} {'Norm Ctrl':>10} {'Norm Trat':>10} {'Fold':>7}\n")
            self.textbox_resultados.insert(ctk.END, "─" * 62 + "\n")
            
            for nombre in sorted(set(norm_ctrl.keys()) | set(norm_exp.keys())):
                vc = crudas_ctrl.get(nombre, 0)
                ve = crudas_exp.get(nombre, 0)
                nc = norm_ctrl.get(nombre, 0)
                ne = norm_exp.get(nombre, 0)
                if nc and nc > 0.001:
                    fold = round(ne / nc, 4) if ne else 0.0
                else:
                    fold = "∞" if ne else "-"
                self.textbox_resultados.insert(ctk.END, 
                    f"{nombre:<10} {vc:>11.2f} {ve:>11.2f} {nc:>10.4f} {ne:>10.4f} {str(fold):>7}\n")
            
        except Exception as e:
            error_mensaje = traceback.format_exc()
            self.textbox_resultados.insert(ctk.END, "\n[X] Ocurrió un error inesperado durante el análisis:\n\n")
            self.textbox_resultados.insert(ctk.END, error_mensaje)
            messagebox.showerror("Error en Análisis", f"Revise el Textbox para leer la traza completa.\nError: {str(e)}")


# Ejecución de la ventana principal
if __name__ == "__main__":
    app = AplicacionApoptosis()
    app.mainloop()
