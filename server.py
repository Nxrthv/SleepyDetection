from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
from io import BytesIO

app = FastAPI()
model = YOLO("Roboflow/best.pt")

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)[0]
    
    alert = "ninguna"
    for box in results.boxes:
        cls = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if cls == "somnolencia" and conf > 0.6:
            alert = "¡Alerta de somnolencia!"
            break

    return JSONResponse(content={"alerta": alert})
