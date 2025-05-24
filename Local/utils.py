import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def guardar_registro(mensaje, ruta_archivo="registro.txt"):
    """Guarda un mensaje en el archivo de registro con fecha y hora"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ruta_archivo, "a") as f:
        f.write(f"[{timestamp}] {mensaje}\n")

def crear_directorios_proyecto():
    """Crea los directorios necesarios para el proyecto"""
    directorios = [
        "modelos",
        "datos",
        "datos/entrenamiento",
        "datos/entrenamiento/despierto",
        "datos/entrenamiento/somnoliento",
        "registros"
    ]
    
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
    
    print("Directorios del proyecto creados correctamente.")

def visualizar_predicciones(modelo, ruta_imagenes, num_imagenes=5):
    """Visualiza las predicciones del modelo en algunas imágenes de prueba"""
    # Cargar imágenes de prueba
    imagenes = []
    etiquetas = []
    
    # Cargar imágenes de clase "despierto"
    ruta_despierto = os.path.join(ruta_imagenes, "despierto")
    archivos_despierto = os.listdir(ruta_despierto)[:num_imagenes]
    
    for archivo in archivos_despierto:
        ruta_completa = os.path.join(ruta_despierto, archivo)
        img = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (96, 96))
        imagenes.append(img)
        etiquetas.append(0)  # 0 = despierto
    
    # Cargar imágenes de clase "somnoliento"
    ruta_somnoliento = os.path.join(ruta_imagenes, "somnoliento")
    archivos_somnoliento = os.listdir(ruta_somnoliento)[:num_imagenes]
    
    for archivo in archivos_somnoliento:
        ruta_completa = os.path.join(ruta_somnoliento, archivo)
        img = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (96, 96))
        imagenes.append(img)
        etiquetas.append(1)  # 1 = somnoliento
    
    # Convertir a arrays numpy
    imagenes = np.array(imagenes)
    etiquetas = np.array(etiquetas)
    
    # Normalizar y preparar para el modelo
    imagenes = imagenes / 255.0
    imagenes = np.expand_dims(imagenes, axis=-1)
    
    # Realizar predicciones
    predicciones = modelo.predict(imagenes)
    predicciones = predicciones.flatten()
    
    # Visualizar resultados
    plt.figure(figsize=(15, 10))
    
    for i in range(len(imagenes)):
        plt.subplot(2, num_imagenes, i + 1)
        plt.imshow(imagenes[i].squeeze(), cmap='gray')
        
        color = 'green' if (predicciones[i] > 0.5) == etiquetas[i] else 'red'
        titulo = f"Real: {'Somnoliento' if etiquetas[i] == 1 else 'Despierto'}\n"
        titulo += f"Pred: {'Somnoliento' if predicciones[i] > 0.5 else 'Despierto'} ({predicciones[i]:.2f})"
        
        plt.title(titulo, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join("modelos", "predicciones.png"))
    plt.show()

def evaluar_modelo(modelo, ruta_datos):
    """Evalúa el modelo en un conjunto de datos y muestra métricas"""
    # Preparar generador de datos
    datagen = ImageDataGenerator(rescale=1./255)
    
    generador = datagen.flow_from_directory(
        ruta_datos,
        target_size=(96, 96),
        batch_size=32,
        color_mode='grayscale',
        class_mode='binary',
        shuffle=False
    )
    
    # Evaluar modelo
    resultados = modelo.evaluate(generador)
    
    print(f"Pérdida: {resultados[0]:.4f}")
    print(f"Precisión: {resultados[1]:.4f}")
    
    # Obtener predicciones
    predicciones = modelo.predict(generador)
    predicciones = (predicciones > 0.5).astype(int).flatten()
    
    # Etiquetas reales
    etiquetas = generador.classes
    
    # Calcular matriz de confusión
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(etiquetas, predicciones)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    
    clases = ['Despierto', 'Somnoliento']
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases, rotation=45)
    plt.yticks(tick_marks, clases)
    
    # Anotar valores en la matriz
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
    plt.savefig(os.path.join("modelos", "matriz_confusion.png"))
    plt.show()
    
    # Mostrar reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(etiquetas, predicciones, target_names=clases))