"""FastAPI entry point for the local Smart Parking ANPR demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from apps.vision.src.recognition.anpr_service import AnprService, AnprSettings
from apps.api.app.services.sensor_service import SensorService

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_ROOT = PROJECT_ROOT / "apps/web"
load_dotenv(PROJECT_ROOT / ".env")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

@lru_cache(maxsize=1)
def get_sensor_service() -> SensorService:
    return SensorService(
        port=os.getenv("ARDUINO_PORT"),
        baudrate=int(os.getenv("ARDUINO_BAUDRATE", "115200")),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sensor = get_sensor_service()
    sensor.start()
    yield
    sensor.stop()


app = FastAPI(title="Smart Parking System", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")


@lru_cache(maxsize=1)
def get_anpr_service() -> AnprService:
    detector_path = Path(
        os.getenv(
            "ANPR_DETECTOR_PATH",
            PROJECT_ROOT / "ml/weights/anpr/license_plate_yolo11n.pt",
        )
    )
    recognizer_dir = Path(
        os.getenv(
            "ANPR_RECOGNIZER_DIR",
            PROJECT_ROOT / "ml/weights/anpr/recognizer",
        )
    )
    return AnprService(
        AnprSettings(
            detector_path=detector_path,
            recognizer_dir=recognizer_dir,
            detection_threshold=float(os.getenv("ANPR_DETECTION_THRESHOLD", "0.40")),
            accept_threshold=float(os.getenv("ANPR_ACCEPT_THRESHOLD", "0.90")),
            review_threshold=float(os.getenv("ANPR_REVIEW_THRESHOLD", "0.75")),
        )
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    try:
        service = get_anpr_service()
        return {
            "status": "ok",
            "models_loaded": True,
            "devices": service.devices,
            "sensor": get_sensor_service().status(),
        }
    except Exception as exc:
        return {"status": "degraded", "models_loaded": False, "detail": str(exc)}


@app.post("/api/anpr")
async def analyze_plate(image: UploadFile = File(...)) -> dict[str, object]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP, or BMP image.")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image size exceeds the 12 MB limit.")
    try:
        return get_anpr_service().analyze_bytes(content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sensor/status")
def sensor_status() -> dict[str, object]:
    return get_sensor_service().status()


@app.post("/api/sensor/simulate")
def simulate_sensor() -> dict[str, object]:
    if os.getenv("APP_ENV", "development") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    sensor = get_sensor_service()
    sensor.simulate_detection()
    return sensor.status()
