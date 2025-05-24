import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import cv2
import time

def crear_modelo():
    """Crea un modelo CNN para detección de somnolencia"""
    modelo = Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Aplanar
        Flatten(),
        
        # Capas densas
        Dense(128, activation='relu'),
        Dropout(0.5),  # Prevenir overfitting
        Dense(1, activation='sigmoid')  # Salida binaria: somnoliento o no
    ])
    
    # Compilar modelo
    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo

def preparar_datos(ruta_datos, tamaño_batch=32, tamaño_img=(96, 96)):
    """Prepara los generadores de datos para entrenamiento y validación"""
    # Aumentación de datos para entrenamiento
    datagen_entrenamiento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% para validación
    )
    
    # Generador para conjunto de entrenamiento
    generador_entrenamiento = datagen_entrenamiento.flow_from_directory(
        ruta_datos,
        target_size=tamaño_img,
        batch_size=tamaño_batch,
        color_mode='grayscale',
        class_mode='binary',
        subset='training'
    )
    
    # Generador para conjunto de validación
    generador_validacion = datagen_entrenamiento.flow_from_directory(
        ruta_datos,
        target_size=tamaño_img,
        batch_size=tamaño_batch,
        color_mode='grayscale',
        class_mode='binary',
        subset='validation'
    )
    
    return generador_entrenamiento, generador_validacion

def entrenar_modelo(ruta_datos, ruta_modelo_salida, epocas=20, tamaño_batch=32):
    """Entrena el modelo con los datos proporcionados"""
    # Crear directorio para modelos si no existe
    os.makedirs(os.path.dirname(ruta_modelo_salida), exist_ok=True)
    
    # Preparar datos
    gen_entrenamiento, gen_validacion = preparar_datos(ruta_datos, tamaño_batch)
    
    # Crear modelo
    modelo = crear_modelo()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        ruta_modelo_salida,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    historia = modelo.fit(
        gen_entrenamiento,
        steps_per_epoch=gen_entrenamiento.samples // tamaño_batch,
        validation_data=gen_validacion,
        validation_steps=gen_validacion.samples // tamaño_batch,
        epochs=epocas,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Guardar modelo final
    modelo.save(ruta_modelo_salida)
    print(f"Modelo guardado en: {ruta_modelo_salida}")
    
    # Graficar resultados
    graficar_historia(historia)
    
    return modelo, historia

def graficar_historia(historia):
    """Grafica la precisión y pérdida del entrenamiento"""
    plt.figure(figsize=(12, 4))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(historia.history['accuracy'], label='Entrenamiento')
    plt.plot(historia.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(historia.history['loss'], label='Entrenamiento')
    plt.plot(historia.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig(os.path.join("modelos", "historia_entrenamiento.png"))
    plt.show()

def recolectar_datos(ruta_salida, num_muestras=100):
    """Función para recolectar datos de entrenamiento usando la cámara"""
    # Crear directorios si no existen
    os.makedirs(os.path.join(ruta_salida, "despierto"), exist_ok=True)
    os.makedirs(os.path.join(ruta_salida, "somnoliento"), exist_ok=True)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara")
        return False
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Recolectar datos para clase "despierto"
    print("Recolectando datos para clase 'despierto'...")
    print("Mantén los ojos abiertos y mira a la cámara")
    input("Presiona Enter para comenzar...")
    
    contador = 0
    while contador < num_muestras:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extraer ROI de la cara
            face_roi = gray[y:y+h, x:x+w]
            
            # Redimensionar
            face_roi = cv2.resize(face_roi, (96, 96))
            
            # Guardar imagen
            cv2.imwrite(os.path.join(ruta_salida, "despierto", f"despierto_{contador}.jpg"), face_roi)
            contador += 1
            
            # Mostrar progreso
            print(f"Despierto: {contador}/{num_muestras}", end="\r")
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.putText(frame, f"Despierto: {contador}/{num_muestras}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Recolección de Datos", frame)
        
        # Esperar tecla
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Esperar un poco entre capturas
        time.sleep(0.1)
    
    # Recolectar datos para clase "somnoliento"
    print("\nRecolectando datos para clase 'somnoliento'...")
    print("Simula estar somnoliento cerrando los ojos o entrecerrándolos")
    input("Presiona Enter para comenzar...")
    
    contador = 0
    while contador < num_muestras:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extraer ROI de la cara
            face_roi = gray[y:y+h, x:x+w]
            
            # Redimensionar
            face_roi = cv2.resize(face_roi, (96, 96))
            
            # Guardar imagen
            cv2.imwrite(os.path.join(ruta_salida, "somnoliento", f"somnoliento_{contador}.jpg"), face_roi)
            contador += 1
            
            # Mostrar progreso
            print(f"Somnoliento: {contador}/{num_muestras}", end="\r")
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.putText(frame, f"Somnoliento: {contador}/{num_muestras}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Recolección de Datos", frame)
        
        # Esperar tecla
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Esperar un poco entre capturas
        time.sleep(0.1)
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nRecolección de datos completada.")
    return True

def main():
    """Función principal"""
    print("=== Entrenamiento de Modelo para Detección de Somnolencia ===")
    print("1. Recolectar datos de entrenamiento")
    print("2. Entrenar modelo con datos existentes")
    print("3. Salir")
    
    opcion = input("Selecciona una opción: ")
    
    if opcion == "1":
        ruta_datos = os.path.join("datos", "entrenamiento")
        num_muestras = int(input("Número de muestras por clase (recomendado: 100-500): ") or "100")
        recolectar_datos(ruta_datos, num_muestras)
    
    elif opcion == "2":
        ruta_datos = input("Ruta a los datos de entrenamiento (default: datos/entrenamiento): ") or "datos/entrenamiento"
        ruta_modelo = os.path.join("modelos", "modelo_somnolencia.h5")
        epocas = int(input("Número de épocas (default: 20): ") or "20")
        
        if not os.path.exists(ruta_datos):
            print(f"Error: La ruta {ruta_datos} no existe.")
            return
        
        entrenar_modelo(ruta_datos, ruta_modelo, epocas)
    
    elif opcion == "3":
        return
    
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()