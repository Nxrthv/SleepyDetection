from ultralytics import YOLO
import cv2

# 1. Cargar el modelo YOLO entrenado
modelo = YOLO('Roboflow/drowsiness_detection/yolov8_entrenamiento/weights/best.pt')  # Reemplaza con la ruta correcta

# 2. Iniciar captura de video (cámara por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("No se pudo acceder a la cámara")

# 3. Bucle de captura y detección en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Realizar predicción con el frame actual
    resultados = modelo(frame)

    # 5. Dibujar los resultados en el frame
    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            nombre_clase = modelo.names[cls]
            label = f'{nombre_clase} {conf:.2f}'

            # Colores según la clase
            if nombre_clase.lower() == "alerta":
                color = (0, 255, 0)
            elif nombre_clase.lower() == "somnoliento":
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 6. Mostrar el resultado en tiempo real
    cv2.imshow('Detección de somnolencia', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Liberar recursos
cap.release()
cv2.destroyAllWindows()
