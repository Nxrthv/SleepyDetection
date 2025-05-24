import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import PIL.Image, PIL.ImageTk
import threading
import time
import os
from detector import SomnolenciaDetector
import numpy as np

class App:
    def __init__(self, window, window_title):
        # Inicializar ventana principal
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x700")
        self.window.configure(bg="#f0f0f0")
        self.window.resizable(width=True, height=True)
        
        # Variables de control
        self.is_running = False
        self.detector = SomnolenciaDetector()
        self.alerta_activa = False
        self.contador_alertas = 0
        self.tiempo_inicio = None
        self.umbral_frames = 15  # Frames consecutivos para detectar somnolencia
        
        # Crear interfaz
        self.crear_widgets()
        
        # Protocolo de cierre
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Iniciar bucle de eventos
        self.window.mainloop()
    
    def crear_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame izquierdo (video)
        self.video_frame = ttk.LabelFrame(main_frame, text="Cámara")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas para mostrar el video
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame derecho (controles)
        control_frame = ttk.LabelFrame(main_frame, text="Controles")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Botones de control
        self.btn_start = ttk.Button(control_frame, text="Iniciar Detección", command=self.iniciar_deteccion)
        self.btn_start.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Detener", command=self.detener_deteccion, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, padx=10, pady=5)
        
        # Separador
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        # Configuración
        config_frame = ttk.LabelFrame(control_frame, text="Configuración")
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Umbral de detección
        ttk.Label(config_frame, text="Umbral de frames:").pack(anchor=tk.W, padx=5, pady=2)
        self.umbral_var = tk.IntVar(value=self.umbral_frames)
        umbral_spin = ttk.Spinbox(config_frame, from_=5, to=30, textvariable=self.umbral_var, width=5)
        umbral_spin.pack(anchor=tk.W, padx=5, pady=2)
        
        # Sensibilidad
        ttk.Label(config_frame, text="Sensibilidad:").pack(anchor=tk.W, padx=5, pady=2)
        self.sensibilidad_var = tk.DoubleVar(value=0.5)
        sensibilidad_scale = ttk.Scale(config_frame, from_=0.1, to=0.9, variable=self.sensibilidad_var, orient=tk.HORIZONTAL)
        sensibilidad_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Separador
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        # Estadísticas
        stats_frame = ttk.LabelFrame(control_frame, text="Estadísticas")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Tiempo de sesión
        ttk.Label(stats_frame, text="Tiempo de sesión:").pack(anchor=tk.W, padx=5, pady=2)
        self.tiempo_label = ttk.Label(stats_frame, text="00:00:00")
        self.tiempo_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Alertas detectadas
        ttk.Label(stats_frame, text="Alertas detectadas:").pack(anchor=tk.W, padx=5, pady=2)
        self.alertas_label = ttk.Label(stats_frame, text="0")
        self.alertas_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Estado actual
        ttk.Label(stats_frame, text="Estado:").pack(anchor=tk.W, padx=5, pady=2)
        self.estado_label = ttk.Label(stats_frame, text="Inactivo", foreground="gray")
        self.estado_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Separador
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        # Botones adicionales
        self.btn_entrenar = ttk.Button(control_frame, text="Entrenar Modelo", command=self.abrir_entrenamiento)
        self.btn_entrenar.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_cargar = ttk.Button(control_frame, text="Cargar Modelo", command=self.cargar_modelo)
        self.btn_cargar.pack(fill=tk.X, padx=10, pady=5)
        
        # Barra de estado
        self.status_bar = ttk.Label(self.window, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def iniciar_deteccion(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo acceder a la cámara")
                return
            
            self.is_running = True
            self.tiempo_inicio = time.time()
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.estado_label.config(text="Activo", foreground="green")
            self.status_bar.config(text="Detección en curso...")
            
            # Iniciar hilo para actualizar tiempo
            self.tiempo_thread = threading.Thread(target=self.actualizar_tiempo)
            self.tiempo_thread.daemon = True
            self.tiempo_thread.start()
            
            # Iniciar captura de video
            self.actualizar_frame()
    
    def detener_deteccion(self):
        self.is_running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.estado_label.config(text="Inactivo", foreground="gray")
        self.status_bar.config(text="Detección detenida")
    
    def actualizar_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Procesar frame con el detector
                frame, somnoliento, confianza = self.detector.procesar_frame(frame, self.sensibilidad_var.get())
                
                # Actualizar estado
                if somnoliento:
                    self.alerta_activa = True
                    self.estado_label.config(text="¡SOMNOLENCIA!", foreground="red")
                    
                    # Incrementar contador de alertas solo una vez por episodio
                    if not hasattr(self, 'ultima_alerta') or time.time() - self.ultima_alerta > 3:
                        self.contador_alertas += 1
                        self.alertas_label.config(text=str(self.contador_alertas))
                        self.ultima_alerta = time.time()
                else:
                    if self.alerta_activa:
                        self.estado_label.config(text="Activo", foreground="green")
                        self.alerta_activa = False
                
                # Convertir frame para mostrar en Tkinter
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(cv2image)
                
                # Redimensionar para ajustar al canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Asegurarse de que el canvas tenga tamaño
                    img_ratio = img.width / img.height
                    canvas_ratio = canvas_width / canvas_height
                    
                    if img_ratio > canvas_ratio:
                        new_width = canvas_width
                        new_height = int(canvas_width / img_ratio)
                    else:
                        new_height = canvas_height
                        new_width = int(canvas_height * img_ratio)
                    
                    img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
                
                self.photo = PIL.ImageTk.PhotoImage(image=img)
                
                # Actualizar canvas
                self.canvas.config(width=img.width, height=img.height)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                # Mostrar confianza
                self.status_bar.config(text=f"Confianza: {confianza:.2f}")
            
            # Programar siguiente actualización
            self.window.after(15, self.actualizar_frame)
    
    def actualizar_tiempo(self):
        while self.is_running:
            if self.tiempo_inicio:
                tiempo_transcurrido = int(time.time() - self.tiempo_inicio)
                horas = tiempo_transcurrido // 3600
                minutos = (tiempo_transcurrido % 3600) // 60
                segundos = tiempo_transcurrido % 60
                tiempo_str = f"{horas:02d}:{minutos:02d}:{segundos:02d}"
                
                # Actualizar etiqueta de tiempo (thread-safe)
                self.window.after(0, lambda: self.tiempo_label.config(text=tiempo_str))
            
            time.sleep(1)
    
    def abrir_entrenamiento(self):
        # Aquí se abriría una ventana para entrenar el modelo
        messagebox.showinfo("Entrenamiento", "Para entrenar el modelo, ejecuta el script 'entrenamiento.py'")
    
    def cargar_modelo(self):
        ruta_modelo = filedialog.askopenfilename(
            title="Seleccionar Modelo",
            filetypes=[("Modelo H5", "*.h5"), ("Todos los archivos", "*.*")]
        )
        
        if ruta_modelo:
            try:
                self.detector.cargar_modelo(ruta_modelo)
                messagebox.showinfo("Éxito", f"Modelo cargado correctamente: {os.path.basename(ruta_modelo)}")
                self.status_bar.config(text=f"Modelo cargado: {os.path.basename(ruta_modelo)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el modelo: {str(e)}")
    
    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
            self.detener_deteccion()
            self.window.destroy()

if __name__ == "__main__":
    app = App(tk.Tk(), "Detector de Somnolencia con IA")