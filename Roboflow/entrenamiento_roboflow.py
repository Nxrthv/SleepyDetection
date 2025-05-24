from ultralytics import YOLO

# Carga el modelo YOLOv8 nano preentrenado
modelo = YOLO("yolov8n.pt")

# Entrenamiento del modelo con los datos personalizados
modelo.train(
    data="datos/roboflow/data.yaml",  # Ruta al archivo YAML de configuración del dataset
    epochs=20,                        # Número de épocas
    imgsz=640,                        # Tamaño de imagen
    batch=8,                          # Tamaño del batch
    name="yolov8_entrenamiento",     # Nombre del proyecto
    project="drowsiness_detection",  # Carpeta donde se guardarán los resultados
    device="cpu",                    # Cambia a "0" si tienes GPU disponible
    workers=0,                  # Ajusta a 0 para evitar problemas en Windows
)