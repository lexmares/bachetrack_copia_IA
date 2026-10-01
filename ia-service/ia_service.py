import io
import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

CONF = float(os.getenv("IA_CONF", "0.45"))

print("Cargando modelo...")
ruta = hf_hub_download("Logesshhh/road-anomaly-pothole-yolov8m", "potbot_yolov8m.pt")
model = YOLO(ruta)
# Calentamiento: la primera predicción siempre es lenta, mejor hacerla al arrancar
model.predict(Image.new("RGB", (640, 640)), verbose=False)
print("Modelo listo")

app = FastAPI(title="BacheTrack IA")


@app.get("/salud")
def salud():
    return {"ok": True}


@app.post("/detectar")
async def detectar(foto: UploadFile = File(...), conf: float = CONF):
    datos = await foto.read()
    try:
        img = Image.open(io.BytesIO(datos))
        # Las fotos de celular traen la rotación en metadatos; esto la aplica
        img = ImageOps.exif_transpose(img).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")

    inicio = time.perf_counter()
    r = model.predict(img, conf=conf, verbose=False)[0]
    tiempo_ms = round((time.perf_counter() - inicio) * 1000)

    alto, ancho = r.orig_shape
    baches = []
    for caja in r.boxes:
        x1, y1, x2, y2 = caja.xyxy[0].tolist()
        baches.append({
            "confianza": round(float(caja.conf), 3),
            "caja": [round(x1), round(y1), round(x2), round(y2)],
            "area_relativa": round((x2 - x1) * (y2 - y1) / (ancho * alto), 4),
        })

    return {
        "hay_bache": len(baches) > 0,
        "total": len(baches),
        "baches": baches,
        "tiempo_ms": tiempo_ms,
    }
