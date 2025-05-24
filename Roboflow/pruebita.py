from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Cargar el modelo entrenado (ajusta la ruta si es diferente)
modelo = YOLO('roboflow/yolov8n.pt')

# 2. Elegir una imagen para predecir (puedes cambiar la ruta a cualquier imagen de prueba)
ruta_imagen = 'img.jpg'

# 3. Realizar la predicción
resultados = modelo(ruta_imagen)

# 4. Mostrar la imagen con las predicciones usando OpenCV
imagen = cv2.imread(ruta_imagen)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Dibujar cajas directamente desde los resultados (opcional si quieres más control)
for r in resultados:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f'{modelo.names[cls]} {conf:.2f}'
        cv2.rectangle(imagen_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(imagen_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 5. Mostrar con matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(imagen_rgb)
plt.axis('off')
plt.title("Predicciones del modelo")
plt.show()