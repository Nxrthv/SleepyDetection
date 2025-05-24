import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

class SomnolenciaDetector:
    def __init__(self, modelo_path=None):
        # Cargar el clasificador de cascada para detección de rostros
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Contador para detección de somnolencia
        self.contador_frames_somnolencia = 0
        self.umbral_frames = 15
        
        # Cargar modelo si se proporciona
        self.modelo = None
        if modelo_path and os.path.exists(modelo_path):
            self.cargar_modelo(modelo_path)
        else:
            # Intentar cargar modelo por defecto
            default_model = os.path.join("modelos", "modelo_somnolencia.h5")
            if os.path.exists(default_model):
                self.cargar_modelo(default_model)
            else:
                print("No se encontró un modelo. Usando detección básica.")
    
    def cargar_modelo(self, modelo_path):
        """Carga un modelo de red neuronal desde un archivo .h5"""
        try:
            self.modelo = load_model(modelo_path)
            print(f"Modelo cargado desde: {modelo_path}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return False
    
    def procesar_frame(self, frame, sensibilidad=0.5):
        """
        Procesa un frame para detectar somnolencia
        
        Args:
            frame: Frame de video a procesar
            sensibilidad: Valor entre 0.1 y 0.9 que ajusta la sensibilidad de detección
            
        Returns:
            frame_procesado: Frame con anotaciones
            somnoliento: Boolean indicando si se detectó somnolencia
            confianza: Valor de confianza de la predicción
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        somnoliento = False
        confianza = 0.0
        
        for (x, y, w, h) in faces:
            # Dibujar rectángulo alrededor de la cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extraer región de interés (ROI) de la cara
            face_roi = gray[y:y+h, x:x+w]
            
            # Si tenemos un modelo de red neuronal, usarlo
            if self.modelo is not None:
                # Preprocesar imagen para la red neuronal
                img = cv2.resize(face_roi, (96, 96))
                img = img / 255.0  # Normalizar
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=-1)  # Añadir canal para escala de grises
                
                # Predecir
                pred = self.modelo.predict(img, verbose=0)[0][0]
                confianza = pred
                
                # Ajustar umbral según sensibilidad
                umbral_ajustado = 0.5 + (sensibilidad - 0.5) * 0.8
                
                if pred > umbral_ajustado:
                    self.contador_frames_somnolencia += 1
                    if self.contador_frames_somnolencia >= self.umbral_frames:
                        somnoliento = True
                        cv2.putText(frame, "SOMNOLENCIA DETECTADA", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.contador_frames_somnolencia = max(0, self.contador_frames_somnolencia - 1)
                
                # Mostrar confianza
                cv2.putText(frame, f"Confianza: {pred:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                # Método alternativo basado en detección de ojos
                eyes = self.eye_cascade.detectMultiScale(face_roi)
                
                if len(eyes) < 2:
                    self.contador_frames_somnolencia += 1
                    if self.contador_frames_somnolencia >= self.umbral_frames:
                        somnoliento = True
                        cv2.putText(frame, "SOMNOLENCIA DETECTADA", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.contador_frames_somnolencia = max(0, self.contador_frames_somnolencia - 1)
                    
                    # Dibujar ojos
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Estimar confianza basada en número de ojos detectados
                confianza = 1.0 - min(len(eyes) * 0.5, 1.0)
        
        # Mostrar contador de frames
        cv2.putText(frame, f"Contador: {self.contador_frames_somnolencia}/{self.umbral_frames}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame, somnoliento, confianza