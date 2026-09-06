import tkinter as tk
from tkinter import ttk, messagebox
import socket
import threading
import csv
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.base import clone
import warnings

# Silenciar advertencias de consola
warnings.filterwarnings('ignore')

# Librerías de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# Configuración del ESP32
ESP_IP = '192.168.4.1'
ESP_PORT = 8080

class PlataformaDiagnosticoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("➕ AutoMed - Detección Simultánea Multi-Output")
        self.root.geometry("1300x750")
        
        # Paleta de colores
        self.bg_negro = "#121212"      
        self.fg_blanco = "#FFFFFF"     
        self.amarillo = "#F1C40F"      
        self.hover_bg = "#2a2a2a"
        self.color_ok = "#2ecc71"
        self.color_warning = "#f39c12"
        self.color_danger = "#e74c3c"
        
        self.root.configure(bg=self.bg_negro)

        # Variables de estado
        self.offsets = [0.0] * 22 
        self.datos_pendientes = [] 
        self.factor_newtons = 0.000009136
        
        # Variables de Machine Learning
        self.modelos_ya_entrenados = False
        self.modelos_entrenados = {}
        self.nombre_mejor_modelo = ""
        self.metricas_ganador = {}
        
        # ARQUITECTURA MULTI-OUTPUT: 3 Modelos de IA puros independientes
        self.ia_suspension = None
        self.ia_frenos = None
        self.ia_alineacion = None

        # Índices de columnas para aislar los subsistemas
        # 0:FPI, 1:SPI, 2:AI, 3:SFI, 4:FFI, 5:FPD, 6:SPD, 7:AD, 8:SFD, 9:FFD
        # 10-21: Acelerómetros (XYZ)
        self.cols_susp = [1, 3, 6, 8, 12, 15, 18, 21] # Celdas Fz + Accel Z
        self.cols_fren = [0, 4, 5, 9, 10, 13, 16, 19] # Celdas Fx + Accel X
        self.cols_alin = [2, 7, 11, 14, 17, 20]       # Celdas Fy + Accel Y

        self.contenedor_principal = tk.Frame(self.root, bg=self.bg_negro)
        self.contenedor_principal.pack(expand=True, fill="both")

        self.mostrar_menu_principal()

    def limpiar_contenedor(self):
        for widget in self.contenedor_principal.winfo_children():
            widget.destroy()

    def crear_boton_volver(self, parent):
        return tk.Button(parent, text="← Volver al menú", font=("Segoe UI", 12, "bold"), bg=self.hover_bg, fg=self.amarillo, activebackground="#3a3a3a", activeforeground=self.amarillo, command=self.mostrar_menu_principal, bd=0, cursor="hand2", padx=15, pady=5)

    # ==========================================
    # 1. PANTALLA: MENÚ PRINCIPAL
    # ==========================================
    def mostrar_menu_principal(self):
        self.limpiar_contenedor()
        wrapper_centro = tk.Frame(self.contenedor_principal, bg=self.bg_negro)
        wrapper_centro.place(relx=0.5, rely=0.45, anchor="center")

        tk.Label(wrapper_centro, text="AutoMed", font=("Segoe UI", 55, "bold"), bg=self.bg_negro, fg=self.amarillo).pack(pady=(0, 5))
        tk.Label(wrapper_centro, text="Suspensión, alineación y frenos", font=("Segoe UI", 18), bg=self.bg_negro, fg=self.fg_blanco).pack(pady=(0, 50))

        frame_botones = tk.Frame(wrapper_centro, bg=self.bg_negro)
        frame_botones.pack()

        def crear_btn(icono, texto, comando):
            btn_frame = tk.Frame(frame_botones, bg=self.bg_negro, highlightbackground=self.amarillo, highlightthickness=2, width=260, height=260)
            btn_frame.pack_propagate(False)
            lbl_icono = tk.Label(btn_frame, text=icono, font=("Segoe UI", 70), bg=self.bg_negro, fg=self.amarillo)
            lbl_icono.pack(expand=True, pady=(20, 0))
            lbl_texto = tk.Label(btn_frame, text=texto, font=("Segoe UI", 16, "bold"), bg=self.bg_negro, fg=self.fg_blanco)
            lbl_texto.pack(expand=True, pady=(0, 20))
            
            def on_enter(e):
                btn_frame.config(bg=self.hover_bg)
                lbl_icono.config(bg=self.hover_bg)
                lbl_texto.config(bg=self.hover_bg)
            def on_leave(e):
                btn_frame.config(bg=self.bg_negro)
                lbl_icono.config(bg=self.bg_negro)
                lbl_texto.config(bg=self.bg_negro)
                
            for w in (btn_frame, lbl_icono, lbl_texto):
                w.bind("<Enter>", on_enter)
                w.bind("<Leave>", on_leave)
                w.bind("<Button-1>", lambda e: comando())
            return btn_frame

        crear_btn("🔍", "Adquisición\ny supervisión", self.mostrar_modulo_adquisicion).pack(side="left", padx=20)
        crear_btn("📈", "Curvas de\ncaracterización", self.mostrar_modulo_curvas).pack(side="left", padx=20)
        crear_btn("🔧", "Diagnóstico\ninteligente", self.iniciar_modulo_diagnostico).pack(side="left", padx=20)

        tk.Label(self.contenedor_principal, text="v6.0 - Arquitectura Multi-Output Independiente Activa", font=("Segoe UI", 10), bg=self.bg_negro, fg="#555555").pack(side="bottom", pady=20)

    # ==========================================
    # 2. MÓDULO: ADQUISICIÓN Y SUPERVISIÓN
    # ==========================================
    def mostrar_modulo_adquisicion(self):
        self.limpiar_contenedor()
        header = tk.Frame(self.contenedor_principal, bg=self.amarillo, height=60)
        header.pack(fill="x")
        tk.Label(header, text="🔍 Módulo de adquisición y supervisión", font=("Segoe UI", 18, "bold"), bg=self.amarillo, fg="#000000").pack(side="left", padx=20, pady=15)
        self.crear_boton_volver(header).pack(side="right", padx=20, pady=10)

        cuerpo = tk.Frame(self.contenedor_principal, bg=self.bg_negro)
        cuerpo.pack(expand=True, fill="both", padx=20, pady=20)
        
        panel_izq = tk.Frame(cuerpo, bg=self.hover_bg, width=320, highlightbackground=self.amarillo, highlightthickness=1)
        panel_izq.pack(side="left", fill="y", padx=(0, 20))
        panel_izq.pack_propagate(False)

        tk.Label(panel_izq, text="Metadatos del ensayo", font=("Segoe UI", 12, "bold"), bg=self.hover_bg, fg=self.amarillo).pack(pady=(15, 10))
        tk.Label(panel_izq, text="Subsistema:", font=("Segoe UI", 10), bg=self.hover_bg, fg=self.fg_blanco).pack(anchor="w", padx=20)
        self.cb_subsistema = ttk.Combobox(panel_izq, values=["Suspensión", "Frenado", "Alineación"], state="readonly", font=("Segoe UI", 10))
        self.cb_subsistema.pack(fill="x", padx=20, pady=(0, 10))
        
        tk.Label(panel_izq, text="Estado inducido:", font=("Segoe UI", 10), bg=self.hover_bg, fg=self.fg_blanco).pack(anchor="w", padx=20)
        self.cb_estado = ttk.Combobox(panel_izq, state="readonly", font=("Segoe UI", 10))
        self.cb_estado.pack(fill="x", padx=20, pady=(0, 10))
        
        def actualizar_etiquetas(e=None):
            sub = self.cb_subsistema.get()
            if sub == "Suspensión": self.cb_estado.config(values=["Normal", "Deterioro moderado", "Deterioro severo"])
            elif sub == "Frenado": self.cb_estado.config(values=["Normal", "Desbalance moderado", "Desbalance severo"])
            elif sub == "Alineación": self.cb_estado.config(values=["Correcta (0°)", "Convergencia (+1°)", "Convergencia (+2°)", "Divergencia (-1°)", "Divergencia (-2°)"])
            self.cb_estado.current(0)
            
        self.cb_subsistema.bind("<<ComboboxSelected>>", actualizar_etiquetas)
        self.cb_subsistema.set("Suspensión")
        actualizar_etiquetas()

        tk.Label(panel_izq, text="Repetición (N°):", font=("Segoe UI", 10), bg=self.hover_bg, fg=self.fg_blanco).pack(anchor="w", padx=20)
        self.var_rep = tk.StringVar(value="1")
        tk.Entry(panel_izq, textvariable=self.var_rep, font=("Segoe UI", 12), justify="center").pack(fill="x", padx=20, pady=(0, 15))

        self.btn_ping = tk.Button(panel_izq, text="Verificar conexión WiFi", font=("Segoe UI", 10, "bold"), bg="#34495e", fg=self.fg_blanco, command=self.verificar_conexion)
        self.btn_ping.pack(fill="x", padx=20, pady=(0, 10))
        self.btn_encerar = tk.Button(panel_izq, text="1. Encerar (1s)", font=("Segoe UI", 12, "bold"), bg="#555555", fg=self.fg_blanco, command=self.ejecutar_encerado)
        self.btn_encerar.pack(fill="x", padx=20, pady=5)
        self.btn_adquirir = tk.Button(panel_izq, text="2. Iniciar ensayo (5s)", font=("Segoe UI", 12, "bold"), bg=self.amarillo, fg="#000000", command=self.ejecutar_adquisicion)
        self.btn_adquirir.pack(fill="x", padx=20, pady=5)

        self.frame_guardado = tk.Frame(panel_izq, bg=self.hover_bg)
        self.btn_guardar = tk.Button(self.frame_guardado, text="✔ Guardar", font=("Segoe UI", 11, "bold"), bg="#27ae60", fg="#ffffff", command=self.guardar_ensayo, width=12)
        self.btn_guardar.pack(side="left", padx=5)
        self.btn_descartar = tk.Button(self.frame_guardado, text="✖ Descartar", font=("Segoe UI", 11, "bold"), bg="#c0392b", fg="#ffffff", command=self.descartar_ensayo, width=12)
        self.btn_descartar.pack(side="right", padx=5)
        self.lbl_status = tk.Label(panel_izq, text="Estado: Listo", font=("Segoe UI", 10, "bold"), bg=self.hover_bg, fg="#3498db")
        self.lbl_status.pack(pady=10)

        panel_der = tk.Frame(cuerpo, bg=self.bg_negro)
        panel_der.pack(side="right", expand=True, fill="both")
        self.fig_adq, self.ax_adq = plt.subplots(figsize=(8, 5))
        self.fig_adq.patch.set_facecolor(self.bg_negro)
        self.ax_adq.set_facecolor(self.hover_bg)
        self.ax_adq.tick_params(colors=self.fg_blanco)
        self.canvas_adq = FigureCanvasTkAgg(self.fig_adq, master=panel_der)
        self.canvas_adq.get_tk_widget().pack(expand=True, fill="both")

    def verificar_conexion(self):
        self.lbl_status.config(text="Buscando ESP32...", fg=self.amarillo)
        def ping():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    s.connect((ESP_IP, ESP_PORT))
                self.root.after(0, lambda: self.lbl_status.config(text="ESP32 Conectado ✔", fg="#2ecc71"))
            except:
                self.root.after(0, lambda: self.lbl_status.config(text="Error: ESP32 Desconectado", fg="#e74c3c"))
        threading.Thread(target=ping, daemon=True).start()

    def comunicar_esp32(self, comando, callback):
        def tarea():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(10.0)
                    s.connect((ESP_IP, ESP_PORT))
                    s.sendall(comando.encode())
                    brutos = ""
                    limpios = []
                    while True:
                        paquete = s.recv(4096).decode('utf-8')
                        if not paquete: break
                        brutos += paquete
                        if "FIN" in brutos:
                            lineas = brutos.replace("FIN\r\n", "").replace("FIN\n", "").replace("FIN", "").split('\n')
                            for linea in lineas:
                                valores = [val for val in linea.strip().split(',') if val]
                                if len(valores) == 22: limpios.append([float(v) for v in valores])
                            break
                self.root.after(0, callback, True, limpios)
            except Exception as e:
                self.root.after(0, callback, False, str(e))
        threading.Thread(target=tarea, daemon=True).start()

    def ejecutar_encerado(self):
        self.lbl_status.config(text="Encerando...", fg=self.amarillo)
        self.comunicar_esp32("E\n", self.fin_encerado)

    def fin_encerado(self, exito, resultado):
        if exito and len(resultado) > 0:
            self.offsets = [sum(col) / len(resultado) for col in zip(*resultado)]
            self.lbl_status.config(text="Sensores encerados en 0.0 ✔", fg="#2ecc71")
        else:
            self.lbl_status.config(text="Error al encerar", fg="#e74c3c")

    def ejecutar_adquisicion(self):
        self.frame_guardado.pack_forget() 
        self.lbl_status.config(text="Adquiriendo datos (5s)...", fg=self.amarillo)
        self.comunicar_esp32("A\n", self.fin_adquisicion)

    def fin_adquisicion(self, exito, resultado):
        if exito and len(resultado) > 0:
            self.datos_pendientes = [[val - off for val, off in zip(fila, self.offsets)] for fila in resultado]
            self.graficar_resultados_adq()
            self.lbl_status.config(text="Ensayo completado. ¿Guardar?", fg=self.amarillo)
            self.frame_guardado.pack(fill="x", padx=20, pady=10)
        else:
            self.lbl_status.config(text="Error de adquisición", fg="#e74c3c")

    def guardar_ensayo(self):
        os.makedirs("dataset", exist_ok=True)
        sub = self.cb_subsistema.get().replace(" ", "_").lower()
        est = self.cb_estado.get().replace(" ", "_").replace("(", "").replace(")", "").replace("°", "").replace("+", "").lower()
        rep = self.var_rep.get()
        nombre = f"dataset/{sub}_{est}_rep{rep}_{datetime.now().strftime('%H%M%S')}.csv"
        with open(nombre, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["FPI", "SPI", "AI", "SFI", "FFI", "FPD", "SPD", "AD", "SFD", "FFD", "API_X", "API_Y", "API_Z", "APD_X", "APD_Y", "APD_Z", "AFD_X", "AFD_Y", "AFD_Z", "AFI_X", "AFI_Y", "AFI_Z"])
            writer.writerows(self.datos_pendientes)
        self.var_rep.set(str(int(rep) + 1))
        self.lbl_status.config(text="Guardado ✔", fg="#2ecc71")
        self.frame_guardado.pack_forget()

    def descartar_ensayo(self):
        self.lbl_status.config(text="Descartado.", fg="#e74c3c")
        self.frame_guardado.pack_forget()

    def graficar_resultados_adq(self):
        self.ax_adq.clear()
        sub = self.cb_subsistema.get()
        datos = self.datos_pendientes
        if sub == "Suspensión":
            self.ax_adq.plot([f[3] for f in datos], label="Frontal izq (SFI)")
            self.ax_adq.plot([f[8] for f in datos], label="Frontal der (SFD)")
            self.ax_adq.plot([f[1] for f in datos], label="Posterior izq (SPI)")
            self.ax_adq.plot([f[6] for f in datos], label="Posterior der (SPD)")
        elif sub == "Frenado":
            self.ax_adq.plot([f[4] for f in datos], label="Frontal izq (FFI)")
            self.ax_adq.plot([f[9] for f in datos], label="Frontal der (FFD)")
            self.ax_adq.plot([f[0] for f in datos], label="Posterior izq (FPI)")
            self.ax_adq.plot([f[5] for f in datos], label="Posterior der (FPD)")
        elif sub == "Alineación":
            self.ax_adq.plot([f[2] for f in datos], label="Lateral izq (AI)")
            self.ax_adq.plot([f[7] for f in datos], label="Lateral der (AD)")

        self.ax_adq.set_title(f"Validación - {sub}", color=self.amarillo)
        self.ax_adq.legend(facecolor=self.bg_negro, edgecolor=self.amarillo, labelcolor=self.fg_blanco)
        self.ax_adq.grid(True, color="#444444", linestyle="--", alpha=0.5)
        self.canvas_adq.draw()

    # ==========================================
    # 3. MÓDULO: CURVAS DE CARACTERIZACIÓN 
    # ==========================================
    def aplicar_filtro_butterworth(self, datos, fs=80, fc=20, orden=4):
        nyq = 0.5 * fs 
        b, a = butter(orden, fc / nyq, btype='low', analog=False)
        return filtfilt(b, a, datos)

    def mostrar_modulo_curvas(self):
        self.limpiar_contenedor()
        header = tk.Frame(self.contenedor_principal, bg=self.amarillo, height=60)
        header.pack(fill="x")
        tk.Label(header, text="📈 Curvas de caracterización", font=("Segoe UI", 18, "bold"), bg=self.amarillo, fg="#000000").pack(side="left", padx=20, pady=15)
        self.crear_boton_volver(header).pack(side="right", padx=20, pady=10)

        cuerpo = tk.Frame(self.contenedor_principal, bg=self.bg_negro)
        cuerpo.pack(expand=True, fill="both", padx=20, pady=20)
        
        panel_izq = tk.Frame(cuerpo, bg=self.hover_bg, width=320, highlightbackground=self.amarillo, highlightthickness=1)
        panel_izq.pack(side="left", fill="y", padx=(0, 20))
        panel_izq.pack_propagate(False)

        tk.Label(panel_izq, text="Análisis paramétrico", font=("Segoe UI", 12, "bold"), bg=self.hover_bg, fg=self.amarillo).pack(pady=(20, 15))
        self.btn_generar_curva = tk.Button(panel_izq, text="Generar curva", font=("Segoe UI", 12, "bold"), bg=self.amarillo, fg="#000000", command=self.procesar_curvas_alineacion)
        self.btn_generar_curva.pack(fill="x", padx=20, pady=10)
        self.lbl_curva_status = tk.Label(panel_izq, text="", font=("Segoe UI", 10), bg=self.hover_bg, fg="#3498db")
        self.lbl_curva_status.pack(pady=20)

        panel_der = tk.Frame(cuerpo, bg=self.bg_negro)
        panel_der.pack(side="right", expand=True, fill="both")
        self.fig_curva, self.ax_curva = plt.subplots(figsize=(8, 5))
        self.fig_curva.patch.set_facecolor(self.bg_negro)
        self.ax_curva.set_facecolor(self.hover_bg)
        self.ax_curva.tick_params(colors=self.fg_blanco)
        self.canvas_curva = FigureCanvasTkAgg(self.fig_curva, master=panel_der)
        self.canvas_curva.get_tk_widget().pack(expand=True, fill="both")

    def procesar_curvas_alineacion(self):
        if not os.path.exists("dataset"): return
        archivos = [f for f in os.listdir("dataset") if f.endswith('.csv') and 'alineaci' in f.lower()]
        self.lbl_curva_status.config(text=f"Analizando {len(archivos)} archivos...", fg=self.amarillo)
        self.root.update()
        
        datos_por_angulo = {-2: {'AI': [], 'AD': []}, -1: {'AI': [], 'AD': []}, 0: {'AI': [], 'AD': []}, 1: {'AI': [], 'AD': []}, 2: {'AI': [], 'AD': []}}
        for archivo in archivos:
            angulo = None
            if "divergencia_-2" in archivo.lower(): angulo = -2
            elif "divergencia_-1" in archivo.lower(): angulo = -1
            elif "correcta_0" in archivo.lower(): angulo = 0
            elif "convergencia_1" in archivo.lower(): angulo = 1
            elif "convergencia_2" in archivo.lower(): angulo = 2
            
            if angulo is not None:
                with open(os.path.join("dataset", archivo), mode='r') as file:
                    reader = csv.reader(file)
                    next(reader) 
                    f_ai, f_ad = [], []
                    for row in reader:
                        if len(row) == 22:
                            f_ai.append(abs(float(row[2])))
                            f_ad.append(abs(float(row[7])))
                    if f_ai and f_ad:
                        datos_por_angulo[angulo]['AI'].append(max(self.aplicar_filtro_butterworth(f_ai)))
                        datos_por_angulo[angulo]['AD'].append(max(self.aplicar_filtro_butterworth(f_ad)))
                        
        angulos_x = sorted(datos_por_angulo.keys())
        p_ai = [np.mean(datos_por_angulo[a]['AI']) * self.factor_newtons if datos_por_angulo[a]['AI'] else 0 for a in angulos_x]
        p_ad = [np.mean(datos_por_angulo[a]['AD']) * self.factor_newtons if datos_por_angulo[a]['AD'] else 0 for a in angulos_x]
        
        self.ax_curva.clear()
        self.ax_curva.plot(angulos_x, p_ai, marker='o', label="Fuerza izq (AI)", color="#2ecc71", linewidth=2)
        self.ax_curva.plot(angulos_x, p_ad, marker='s', label="Fuerza der (AD)", color="#1abc9c", linewidth=2)
        self.ax_curva.set_title("Caracterización geométrica", color=self.amarillo)
        self.ax_curva.set_xticks(angulos_x)
        self.ax_curva.grid(True, color="#444444", linestyle="--", alpha=0.5)
        self.ax_curva.legend(facecolor=self.bg_negro, labelcolor=self.fg_blanco)
        self.canvas_curva.draw()
        self.lbl_curva_status.config(text="Curvas trazadas ✔", fg="#2ecc71")

    # ==========================================
    # 4. MÓDULO: DIAGNÓSTICO INTELIGENTE (MULTI-OUTPUT REAL)
    # ==========================================
    def extraer_caracteristicas_desde_archivo(self, ruta_csv):
        try:
            datos = np.loadtxt(ruta_csv, delimiter=',', skiprows=1)
            return self.extraer_caracteristicas_de_matriz(datos)
        except: return None

    def extraer_caracteristicas_de_matriz(self, datos):
        datos = np.array(datos)
        if datos.shape[1] != 22: return None
        
        datos_filtrados = np.zeros_like(datos)
        for col in range(22):
            try:
                datos_filtrados[:, col] = self.aplicar_filtro_butterworth(datos[:, col])
            except:
                datos_filtrados[:, col] = datos[:, col] 
                
        energia_instantanea = np.sum(np.abs(datos_filtrados), axis=1)
        umbral_ruido = np.max(energia_instantanea) * 0.20
        indices_evento = np.where(energia_instantanea > umbral_ruido)[0]
        
        if len(indices_evento) > 5:
            datos_limpios = datos_filtrados[indices_evento]
        else:
            datos_limpios = datos_filtrados 
            
        features = []
        for col in range(22):
            col_data = datos_limpios[:, col]
            features.extend([
                np.sqrt(np.mean(col_data**2)), # RMS
                np.max(col_data),              # Máximo real (mantiene polaridad)
                np.min(col_data),              # Mínimo real (mantiene polaridad)
                np.mean(col_data),             # Media
                np.std(col_data)               # Desviación estándar
            ])
            
        return features

    def filtrar_features(self, features_completos, columnas_deseadas):
        # Cada sensor ahora genera 5 características, extraemos en bloques de 5
        features_filtrados = []
        for col in columnas_deseadas:
            inicio = col * 5
            fin = inicio + 5
            features_filtrados.extend(features_completos[inicio:fin])
        return features_filtrados

    def iniciar_modulo_diagnostico(self):
        if not self.modelos_ya_entrenados:
            self.auto_entrenar_silencioso()
        self.mostrar_dashboard()

    def auto_entrenar_silencioso(self):
        dataset_dir = "dataset"
        if not os.path.exists(dataset_dir): return
        archivos = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        if len(archivos) < 5: return

        X, y_general = [], []
        y_susp, y_fren, y_alin = [], [], []
        
        for archivo in archivos:
            clase = 0 
            if "suspensi" in archivo.lower() and "moderado" in archivo.lower(): clase = 1
            elif "suspensi" in archivo.lower() and "severo" in archivo.lower(): clase = 2
            elif "frenado" in archivo.lower() and "moderado" in archivo.lower(): clase = 3
            elif "frenado" in archivo.lower() and "severo" in archivo.lower(): clase = 4
            elif "alineaci" in archivo.lower() and "convergencia" in archivo.lower(): clase = 5
            elif "alineaci" in archivo.lower() and "divergencia" in archivo.lower(): clase = 6
                
            features = self.extraer_caracteristicas_desde_archivo(os.path.join(dataset_dir, archivo))
            if features:
                X.append(features)
                y_general.append(clase)
                
                # Mapeo para Entrenamiento de Modelos Independientes
                y_susp.append(clase if clase in [1, 2] else 0)
                y_fren.append(1 if clase == 3 else (2 if clase == 4 else 0))
                y_alin.append(1 if clase == 5 else (2 if clase == 6 else 0))
                
        # 1. BÚSQUEDA DEL ALGORITMO GANADOR (Evaluación Tesis)
        X_train, X_test, y_train, y_test = train_test_split(X, y_general, test_size=0.3, random_state=42)

        modelos = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='linear', probability=True, random_state=42),
            "MLP (Red Neuronal)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "Regresión Logística": LogisticRegression(max_iter=5000, random_state=42)
        }

        resultados_acc = {}
        for nombre, clf in modelos.items():
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            resultados_acc[nombre] = accuracy_score(y_test, preds) * 100
            self.modelos_entrenados[nombre] = clf
            
            self.metricas_ganador[nombre] = {
                "acc": accuracy_score(y_test, preds),
                "prec": precision_score(y_test, preds, average='weighted', zero_division=0),
                "rec": recall_score(y_test, preds, average='weighted', zero_division=0),
                "f1": f1_score(y_test, preds, average='weighted', zero_division=0),
                "cm": confusion_matrix(y_test, preds)
            }

        self.nombre_mejor_modelo = max(resultados_acc, key=resultados_acc.get)
        modelo_base = self.modelos_entrenados[self.nombre_mejor_modelo]
        
        # 2. ENTRENAMIENTO MULTI-OUTPUT: 3 IAs expertas aisladas
        X_susp = [self.filtrar_features(f, self.cols_susp) for f in X]
        X_fren = [self.filtrar_features(f, self.cols_fren) for f in X]
        X_alin = [self.filtrar_features(f, self.cols_alin) for f in X]

        self.ia_suspension = clone(modelo_base)
        self.ia_frenos = clone(modelo_base)
        self.ia_alineacion = clone(modelo_base)

        self.ia_suspension.fit(X_susp, y_susp)
        self.ia_frenos.fit(X_fren, y_fren)
        self.ia_alineacion.fit(X_alin, y_alin)

        self.modelos_ya_entrenados = True

    def forzar_reentrenamiento(self):
        messagebox.showinfo("Re-entrenamiento", "El sistema ha re-entrenado las 3 IAs independientes con la arquitectura ganadora.")
        self.modelos_ya_entrenados = False
        self.auto_entrenar_silencioso()
        self.mostrar_evaluacion_modelos()

    def construir_esqueleto_diagnostico(self):
        self.limpiar_contenedor()
        header = tk.Frame(self.contenedor_principal, bg=self.amarillo, height=60)
        header.pack(fill="x")
        tk.Label(header, text="🔧 Diagnóstico con IA Multi-Output", font=("Segoe UI", 18, "bold"), bg=self.amarillo, fg="#000000").pack(side="left", padx=20, pady=15)
        self.crear_boton_volver(header).pack(side="right", padx=20, pady=10)

        self.menu_tabs = tk.Frame(self.contenedor_principal, bg=self.hover_bg, height=50)
        self.menu_tabs.pack(fill="x")
        
        self.btn_tab_dash = tk.Button(self.menu_tabs, text="🚗 Dashboard de diagnóstico", font=("Segoe UI", 12, "bold"), bg="#2980b9", fg=self.fg_blanco, bd=0, command=self.mostrar_dashboard)
        self.btn_tab_dash.pack(side="left", padx=20, pady=10)
        self.btn_tab_eval = tk.Button(self.menu_tabs, text="📊 Información y evaluación de modelos", font=("Segoe UI", 12, "bold"), bg="#34495e", fg=self.fg_blanco, bd=0, command=self.mostrar_evaluacion_modelos)
        self.btn_tab_eval.pack(side="left", padx=10, pady=10)

        self.diag_cuerpo = tk.Frame(self.contenedor_principal, bg=self.bg_negro)
        self.diag_cuerpo.pack(expand=True, fill="both", padx=20, pady=20)

    # --- PESTAÑA A: DASHBOARD DE DIAGNÓSTICO EN VIVO ---
    def mostrar_dashboard(self):
        self.construir_esqueleto_diagnostico()
        self.btn_tab_dash.config(state="disabled", bg="#1f618d") 
        self.btn_tab_eval.config(state="normal", bg="#34495e")

        if not self.ia_suspension:
            tk.Label(self.diag_cuerpo, text="⚠️ Faltan datos en la carpeta 'dataset' para entrenar la IA.", font=("Segoe UI", 16), bg=self.bg_negro, fg=self.color_warning).pack(pady=100)
            return

        panel_izq = tk.Frame(self.diag_cuerpo, bg=self.bg_negro, width=320)
        panel_izq.pack(side="left", fill="y", padx=10)
        panel_izq.pack_propagate(False)

        # ARRIBA: TARJETAS
        self.frame_tarjetas = tk.Frame(panel_izq, bg=self.bg_negro)
        self.frame_tarjetas.pack(fill="x")
        tk.Label(self.frame_tarjetas, text=f"Arquitectura: {self.nombre_mejor_modelo} (Multi-Output)", font=("Segoe UI", 10, "bold"), bg=self.bg_negro, fg=self.amarillo).pack(pady=(0, 10))
        
        self.generar_tarjetas_ui(0, 0, 0) 

        # ABAJO: CONTROLES DE CONEXIÓN
        panel_conexion = tk.Frame(panel_izq, bg=self.hover_bg, highlightbackground=self.amarillo, highlightthickness=1)
        panel_conexion.pack(fill="x", side="bottom", pady=10)
        
        tk.Label(panel_conexion, text="Control ESP32", font=("Segoe UI", 11, "bold"), bg=self.hover_bg, fg=self.amarillo).pack(pady=(10, 5))
        self.btn_ping_diag = tk.Button(panel_conexion, text="Verificar conexión WiFi", font=("Segoe UI", 10, "bold"), bg="#34495e", fg=self.fg_blanco, command=self.verificar_conexion_diag)
        self.btn_ping_diag.pack(fill="x", padx=15, pady=(0, 5))
        self.btn_encerar_diag = tk.Button(panel_conexion, text="1. Encerar (1s)", font=("Segoe UI", 10, "bold"), bg="#555555", fg=self.fg_blanco, command=self.ejecutar_encerado_diag)
        self.btn_encerar_diag.pack(fill="x", padx=15, pady=5)
        self.btn_diagnosticar = tk.Button(panel_conexion, text="2. Diagnosticar (5s)", font=("Segoe UI", 11, "bold"), bg=self.color_ok, fg="#000000", command=self.ejecutar_diagnostico_real)
        self.btn_diagnosticar.pack(fill="x", padx=15, pady=5)
        self.lbl_status_diag = tk.Label(panel_conexion, text="Listo para escanear", font=("Segoe UI", 9, "bold"), bg=self.hover_bg, fg="#3498db")
        self.lbl_status_diag.pack(pady=5)

        # PANEL DERECHO (DINÁMICO: COCHE O CURVAS)
        self.panel_der = tk.Frame(self.diag_cuerpo, bg=self.hover_bg, highlightbackground="#444444", highlightthickness=2)
        self.panel_der.pack(side="right", expand=True, fill="both", padx=(15, 0))
        
        # Sub-panel 1: Vista del Coche
        self.vista_coche = tk.Frame(self.panel_der, bg=self.hover_bg)
        tk.Label(self.vista_coche, text="Localización de daños físicos", font=("Segoe UI", 16, "bold"), bg=self.hover_bg, fg=self.fg_blanco).pack(pady=10)
        self.cv_car = tk.Canvas(self.vista_coche, width=500, height=550, bg=self.hover_bg, highlightthickness=0)
        self.cv_car.pack(expand=True)
        self.vista_coche.pack(expand=True, fill="both")
        self.dibujar_coche_diagnostico(0, 0, 0) 

        # Sub-panel 2: Vista de Curvas (Oculto al inicio)
        self.vista_curvas = tk.Frame(self.panel_der, bg=self.hover_bg)
        top_curvas = tk.Frame(self.vista_curvas, bg=self.hover_bg)
        top_curvas.pack(fill="x", pady=10)
        tk.Label(top_curvas, text="Respuesta dinámica del subsistema", font=("Segoe UI", 16, "bold"), bg=self.hover_bg, fg=self.fg_blanco).pack(side="left", padx=20)
        tk.Button(top_curvas, text="← Volver a vista del vehículo", font=("Segoe UI", 10, "bold"), bg="#c0392b", fg=self.fg_blanco, command=self.volver_a_coche, cursor="hand2").pack(side="right", padx=20)
        
        self.fig_dash, self.ax_dash = plt.subplots(figsize=(7, 4))
        self.fig_dash.patch.set_facecolor(self.hover_bg)
        self.ax_dash.set_facecolor(self.bg_negro)
        self.ax_dash.tick_params(colors=self.fg_blanco)
        self.canvas_dash = FigureCanvasTkAgg(self.fig_dash, master=self.vista_curvas)
        self.canvas_dash.get_tk_widget().pack(expand=True, fill="both")

    def volver_a_coche(self):
        self.vista_curvas.pack_forget()
        self.vista_coche.pack(expand=True, fill="both")

    def generar_tarjetas_ui(self, p_susp, p_fren, p_alin):
        for widget in self.frame_tarjetas.winfo_children()[1:]: widget.destroy() 
        
        def al_clickear_tarjeta(event, subsistema):
            if not self.datos_pendientes:
                messagebox.showinfo("Sin datos", "Realice un diagnóstico (5s) primero para ver las curvas.")
                return
            self.vista_coche.pack_forget()
            self.vista_curvas.pack(expand=True, fill="both")
            self.graficar_curva_dashboard(subsistema)

        def crear_tarjeta(titulo, estado, color, id_subsistema):
            card = tk.Frame(self.frame_tarjetas, bg="#1e272e", highlightbackground="#34495e", highlightthickness=2, height=110, cursor="hand2")
            card.pack(fill="x", pady=7)
            card.pack_propagate(False)
            
            # --- DIBUJO DE ICONOS VECTORIALES ---
            cv_ico = tk.Canvas(card, width=60, height=60, bg="#1e272e", highlightthickness=0, cursor="hand2")
            cv_ico.place(x=15, y=20)
            
            if id_subsistema == "Suspensión":
                cv_ico.create_line(30, 5, 30, 55, fill="#bdc3c7", width=3) 
                cv_ico.create_rectangle(24, 8, 36, 22, fill="#7f8c8d", outline="#95a5a6") 
                cv_ico.create_line(18,22, 42,27, 18,32, 42,37, 18,42, 42,47, 18,52, fill="#e67e22", width=3, smooth=True) 
            elif id_subsistema == "Frenado":
                cv_ico.create_oval(8, 8, 52, 52, fill="#95a5a6", outline="#bdc3c7", width=2) 
                cv_ico.create_oval(22, 22, 38, 38, fill="#34495e", outline="#7f8c8d", width=1) 
                cv_ico.create_arc(10, 10, 50, 50, start=135, extent=90, outline="#e74c3c", width=7, style=tk.ARC) 
            elif id_subsistema == "Alineación":
                cv_ico.create_line(15, 30, 45, 30, fill="#7f8c8d", width=3) 
                cv_ico.create_rectangle(10, 12, 18, 48, fill="#2c3e50", outline="#bdc3c7", width=2) 
                cv_ico.create_rectangle(42, 12, 50, 48, fill="#2c3e50", outline="#bdc3c7", width=2) 
                cv_ico.create_line(5, 20, 10, 25, fill="#e74c3c", width=2, arrow=tk.LAST) 
                cv_ico.create_line(55, 20, 50, 25, fill="#e74c3c", width=2, arrow=tk.LAST) 

            sep = tk.Frame(card, bg="#555555", width=2, height=80)
            sep.place(x=90, y=15)

            lbl_tit = tk.Label(card, text=titulo, font=("Segoe UI", 16, "bold"), bg="#1e272e", fg=self.fg_blanco, cursor="hand2")
            lbl_tit.place(x=105, y=15)
            
            canvas_bar = tk.Canvas(card, width=180, height=12, bg="#111111", highlightthickness=0, cursor="hand2")
            canvas_bar.place(x=105, y=50)
            canvas_bar.create_rectangle(0, 0, 180, 12, fill="#333333", width=0)
            width_bar = 180 if color == self.color_ok else (110 if color == self.color_warning else 50)
            canvas_bar.create_rectangle(0, 0, width_bar, 12, fill=color, width=0)
            
            lbl_est = tk.Label(card, text=estado, font=("Segoe UI", 10, "bold"), bg="#1e272e", fg=color, cursor="hand2")
            lbl_est.place(x=105, y=70)

            for w in [card, cv_ico, lbl_tit, canvas_bar, lbl_est, sep]:
                w.bind("<Button-1>", lambda e, sub=id_subsistema: al_clickear_tarjeta(e, sub))

        # Tarjeta 1: Suspensión
        e_susp, c_susp = "Estado nominal", self.color_ok
        if p_susp == 1: e_susp, c_susp = "Deterioro moderado", self.color_warning
        elif p_susp == 2: e_susp, c_susp = "Deterioro severo", self.color_danger
        crear_tarjeta("Suspensión", e_susp, c_susp, "Suspensión")
        
        # Tarjeta 2: Frenos
        e_fren, c_fren = "Estado nominal", self.color_ok
        if p_fren == 3: e_fren, c_fren = "Desbalance moderado", self.color_warning
        elif p_fren == 4: e_fren, c_fren = "Desbalance severo", self.color_danger
        crear_tarjeta("Frenos", e_fren, c_fren, "Frenado")
        
        # Tarjeta 3: Alineación
        e_alin, c_alin = "Estado normal", self.color_ok
        if p_alin == 5: e_alin, c_alin = "Convergencia", self.color_danger
        elif p_alin == 6: e_alin, c_alin = "Divergencia", self.color_danger
        crear_tarjeta("Alineación", e_alin, c_alin, "Alineación")

    def dibujar_coche_diagnostico(self, p_susp, p_fren, p_alin):
        self.cv_car.delete("all")
        
        # Ejes
        self.cv_car.create_line(150, 140, 350, 140, fill="#555555", width=8) 
        self.cv_car.create_line(150, 380, 350, 380, fill="#555555", width=8) 
        
        # Chasis central
        self.cv_car.create_rectangle(180, 100, 320, 420, fill="#2c3e50", outline=self.amarillo, width=3)
        self.cv_car.create_polygon(180, 200, 320, 200, 300, 160, 200, 160, fill="#1abc9c", outline="#16a085") 
        self.cv_car.create_rectangle(190, 210, 310, 330, fill="#34495e", outline="#2c3e50") 
        self.cv_car.create_text(250, 270, text="AutoMed", fill="#bdc3c7", font=("Segoe UI", 18, "bold"), angle=90)
        
        def dibujar_rueda(x, y, color):
            self.cv_car.create_rectangle(x, y, x+35, y+90, fill="#111111", outline=color, width=4)
            for i in range(15, 90, 15): self.cv_car.create_line(x, y+i, x+35, y+i, fill="#333333", width=2)

        c_front_izq = self.color_ok
        c_front_der = self.color_ok
        c_rear_izq = self.color_ok
        c_rear_der = self.color_ok
        
        # Superposición de colores por fallas independientes
        if p_susp == 1: c_rear_izq = c_rear_der = self.color_warning 
        elif p_susp == 2: c_rear_izq = c_rear_der = self.color_danger 
        
        if p_fren == 3: c_front_der = self.color_warning 
        elif p_fren == 4: c_front_der = self.color_danger 
        
        if p_alin in [5, 6]: 
            c_front_izq = self.color_danger
            c_front_der = self.color_danger

        dibujar_rueda(115, 95, c_front_izq) 
        dibujar_rueda(350, 95, c_front_der) 
        dibujar_rueda(115, 335, c_rear_izq) 
        dibujar_rueda(350, 335, c_rear_der) 

    def graficar_curva_dashboard(self, subsistema):
        self.ax_dash.clear()
        datos = self.datos_pendientes
        if not datos: return
        
        if subsistema == "Suspensión":
            self.ax_dash.plot([f[3] for f in datos], label="Frontal izq (SFI)", color="#3498db")
            self.ax_dash.plot([f[8] for f in datos], label="Frontal der (SFD)", color=self.amarillo)
            self.ax_dash.plot([f[1] for f in datos], label="Posterior izq (SPI)", color="#9b59b6")
            self.ax_dash.plot([f[6] for f in datos], label="Posterior der (SPD)", color="#e67e22")
            self.ax_dash.set_ylabel("Fuerza vertical")
        elif subsistema == "Frenado":
            self.ax_dash.plot([f[4] for f in datos], label="Frontal izq (FFI)", color="#e74c3c")
            self.ax_dash.plot([f[9] for f in datos], label="Frontal der (FFD)", color="#f1c40f")
            self.ax_dash.plot([f[0] for f in datos], label="Posterior izq (FPI)", color="#c0392b")
            self.ax_dash.plot([f[5] for f in datos], label="Posterior der (FPD)", color="#d35400")
            self.ax_dash.set_ylabel("Fuerza longitudinal")
        elif subsistema == "Alineación":
            self.ax_dash.plot([f[2] for f in datos], label="Lateral izq (AI)", color="#2ecc71")
            self.ax_dash.plot([f[7] for f in datos], label="Lateral der (AD)", color="#1abc9c")
            self.ax_dash.set_ylabel("Fuerza lateral")

        self.ax_dash.set_title(f"Señal evaluada por las IAs: {subsistema}", color=self.amarillo)
        self.ax_dash.legend(facecolor=self.bg_negro, labelcolor=self.fg_blanco)
        self.ax_dash.grid(True, color="#444444", linestyle="--", alpha=0.5)
        self.canvas_dash.draw()

    # --- CONTROLES DE CONEXIÓN DIAGNÓSTICO ---
    def verificar_conexion_diag(self):
        self.lbl_status_diag.config(text="Buscando ESP32...", fg=self.amarillo)
        def ping():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    s.connect((ESP_IP, ESP_PORT))
                self.root.after(0, lambda: self.lbl_status_diag.config(text="ESP32 Conectado ✔", fg="#2ecc71"))
            except:
                self.root.after(0, lambda: self.lbl_status_diag.config(text="Error: ESP32 desconectado", fg="#e74c3c"))
        threading.Thread(target=ping, daemon=True).start()

    def ejecutar_encerado_diag(self):
        self.lbl_status_diag.config(text="Encerando...", fg=self.amarillo)
        self.btn_diagnosticar.config(state="disabled")
        self.comunicar_esp32("E\n", self.fin_encerado_diag)

    def fin_encerado_diag(self, exito, resultado):
        self.btn_diagnosticar.config(state="normal")
        if exito and len(resultado) > 0:
            self.offsets = [sum(col) / len(resultado) for col in zip(*resultado)]
            self.lbl_status_diag.config(text="Sensores en 0.0 ✔", fg="#2ecc71")
        else:
            self.lbl_status_diag.config(text="Error al encerar", fg="#e74c3c")

    def ejecutar_diagnostico_real(self):
        self.lbl_status_diag.config(text="Evaluando vehículo...", fg=self.amarillo)
        self.btn_diagnosticar.config(state="disabled")
        self.btn_encerar_diag.config(state="disabled")
        self.comunicar_esp32("A\n", self.fin_diagnostico_real)

    def fin_diagnostico_real(self, exito, resultado):
        self.btn_diagnosticar.config(state="normal")
        self.btn_encerar_diag.config(state="normal")
        
        if exito and len(resultado) > 0:
            self.lbl_status_diag.config(text="Analizando con 3 IAs Paralelas...", fg=self.amarillo)
            self.root.update()
            
            self.datos_pendientes = [[val - off for val, off in zip(fila, self.offsets)] for fila in resultado]
            features = self.extraer_caracteristicas_de_matriz(self.datos_pendientes)
            
            if features:
                # Filtramos las entradas en tiempo real
                f_susp = self.filtrar_features(features, self.cols_susp)
                f_fren = self.filtrar_features(features, self.cols_fren)
                f_alin = self.filtrar_features(features, self.cols_alin)

                # CADA IA EVALÚA SOLO SU SUBSISTEMA
                raw_susp = int(self.ia_suspension.predict([f_susp])[0])
                raw_fren = int(self.ia_frenos.predict([f_fren])[0])
                raw_alin = int(self.ia_alineacion.predict([f_alin])[0])
                
                # Mapeo a las clases originales (0 a 6) para la Interfaz
                pred_susp = raw_susp  # 0, 1, o 2
                pred_fren = {0: 0, 1: 3, 2: 4}.get(raw_fren, 0)
                pred_alin = {0: 0, 1: 5, 2: 6}.get(raw_alin, 0)
                
                self.volver_a_coche() 
                self.generar_tarjetas_ui(pred_susp, pred_fren, pred_alin)
                self.dibujar_coche_diagnostico(pred_susp, pred_fren, pred_alin)
                self.lbl_status_diag.config(text=f"¡Diagnóstico completado! ✔", fg=self.color_ok)
            else:
                self.lbl_status_diag.config(text="Error de extracción", fg="#e74c3c")
        else:
            self.lbl_status_diag.config(text="Fallo de conexión", fg="#e74c3c")

    # --- PESTAÑA B: INFORMACIÓN Y EVALUACIÓN DE MODELOS ---
    def mostrar_evaluacion_modelos(self):
        self.construir_esqueleto_diagnostico()
        self.btn_tab_eval.config(state="disabled", bg="#1f618d") 
        self.btn_tab_dash.config(state="normal", bg="#2980b9")
        
        panel_izq = tk.Frame(self.diag_cuerpo, bg=self.hover_bg, width=280, highlightbackground=self.amarillo, highlightthickness=1)
        panel_izq.pack(side="left", fill="y", padx=(0, 20))
        panel_izq.pack_propagate(False)

        tk.Label(panel_izq, text="Rendimiento de la IA", font=("Segoe UI", 12, "bold"), bg=self.hover_bg, fg=self.amarillo).pack(pady=(15, 5))
        
        btn_reentrenar = tk.Button(panel_izq, text="🔄 Re-entrenar IA", font=("Segoe UI", 10, "bold"), bg="#e67e22", fg=self.fg_blanco, command=self.forzar_reentrenamiento)
        btn_reentrenar.pack(fill="x", padx=15, pady=5)
        
        if self.nombre_mejor_modelo != "":
            tk.Label(panel_izq, text=f"Arquitectura base:\n{self.nombre_mejor_modelo}", font=("Segoe UI", 12, "bold"), bg=self.hover_bg, fg=self.color_ok).pack(pady=10)
            
            frame_metricas = tk.Frame(panel_izq, bg=self.bg_negro, highlightbackground="#444", highlightthickness=1)
            frame_metricas.pack(fill="both", expand=True, padx=15, pady=15)
            tk.Label(frame_metricas, text="Métricas oficiales", font=("Segoe UI", 10, "bold"), bg=self.bg_negro, fg=self.fg_blanco).pack(pady=10)
            
            mets = self.metricas_ganador[self.nombre_mejor_modelo]
            tk.Label(frame_metricas, text=f"Exactitud (Acc): {mets['acc']*100:.2f}%", font=("Segoe UI", 10), bg=self.bg_negro, fg=self.color_ok).pack(anchor="w", padx=10, pady=5)
            tk.Label(frame_metricas, text=f"Precisión: {mets['prec']*100:.2f}%", font=("Segoe UI", 10), bg=self.bg_negro, fg=self.color_warning).pack(anchor="w", padx=10, pady=5)
            tk.Label(frame_metricas, text=f"Sensibilidad: {mets['rec']*100:.2f}%", font=("Segoe UI", 10), bg=self.bg_negro, fg="#3498db").pack(anchor="w", padx=10, pady=5)
            tk.Label(frame_metricas, text=f"F1-Score: {mets['f1']*100:.2f}%", font=("Segoe UI", 10), bg=self.bg_negro, fg="#9b59b6").pack(anchor="w", padx=10, pady=5)

        panel_der_eval = tk.Frame(self.diag_cuerpo, bg=self.bg_negro)
        panel_der_eval.pack(side="right", expand=True, fill="both")
        
        if self.modelos_ya_entrenados:
            fig_ml, (ax_bar, ax_cm) = plt.subplots(1, 2, figsize=(10, 5))
            fig_ml.patch.set_facecolor(self.bg_negro)
            fig_ml.subplots_adjust(bottom=0.3, wspace=0.3)
            canvas_ml = FigureCanvasTkAgg(fig_ml, master=panel_der_eval)
            canvas_ml.get_tk_widget().pack(expand=True, fill="both")

            ax_bar.set_facecolor(self.bg_negro)
            ax_bar.tick_params(colors=self.fg_blanco)
            nombres = list(self.metricas_ganador.keys())
            valores = [self.metricas_ganador[n]["acc"] * 100 for n in nombres]
            colores = [self.color_ok if n == self.nombre_mejor_modelo else "#3498db" for n in nombres]
            
            ax_bar.bar(nombres, valores, color=colores)
            ax_bar.set_title("Comparativa de exactitud", color=self.amarillo, fontsize=12, fontweight="bold")
            ax_bar.set_ylabel("Precisión (%)", color=self.fg_blanco)
            ax_bar.set_ylim(0, 110)
            ax_bar.tick_params(axis='x', rotation=45, labelsize=8)

            ax_cm.set_facecolor(self.bg_negro)
            disp = ConfusionMatrixDisplay(confusion_matrix=self.metricas_ganador[self.nombre_mejor_modelo]['cm'])
            disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
            ax_cm.set_title("Matriz de confusión", color=self.amarillo, fontsize=12, fontweight="bold")
            ax_cm.tick_params(colors=self.fg_blanco)
            ax_cm.xaxis.label.set_color(self.fg_blanco)
            ax_cm.yaxis.label.set_color(self.fg_blanco)

            fig_ml.tight_layout()
            canvas_ml.draw()

if __name__ == "__main__":
    ventana_principal = tk.Tk()
    app = PlataformaDiagnosticoApp(ventana_principal)
    ventana_principal.mainloop()