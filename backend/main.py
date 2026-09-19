"""Ankle ROM API: stateless FastAPI service around an MMPose HRNet-W32 model."""
import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from geometry import draw_skeleton, included_angle, to_base64_jpeg
from pose import PoseEstimator, ensure_weights

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rom.api")

# --- Configuration (override via env vars on Railway) -------------------------------------
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?id=1hKlDm26WiL1rW0EO67DydPNp1BTxvd-E")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/weights/hrnet_w32_ankle.pth")
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "configs/hrnet_w32_ankle.py")  # MMPose config used for training
DEVICE = os.getenv("DEVICE", "cpu")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Ankle ROM API")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def load_model() -> None:
    """Download weights (once) and load the model into memory. Runs at boot, never per request."""
    if MODEL_URL == "DUMMY_GOOGLE_DRIVE_LINK" and not os.path.exists(MODEL_PATH):
        log.warning("MODEL_URL is still a dummy value - starting WITHOUT a model (endpoint returns 503).")
        return
    checkpoint = ensure_weights(MODEL_URL, MODEL_PATH)
    app.state.estimator = PoseEstimator(MODEL_CONFIG, checkpoint, device=DEVICE)
    log.info("Model ready.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": hasattr(app.state, "estimator")}


async def _decode(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename}: file exceeds 10 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{upload.filename}: not a valid image.")
    return img


def _analyse(img: np.ndarray, label: str) -> tuple[float, str]:
    """Run inference on one image -> (angle in degrees, annotated image as base64 JPEG)."""
    try:
        kpts = app.state.estimator.predict(img)
        angle = included_angle(kpts["bottom_heel"], kpts["5th_metatarsal"], kpts["shin"])
    except ValueError as e:
        raise HTTPException(422, f"{label}: {e}")
    return angle, to_base64_jpeg(draw_skeleton(img, kpts, angle))


@app.post("/api/calculate_rom")
async def calculate_rom(image_rest: UploadFile = File(...), image_dorsi: UploadFile = File(...)) -> dict:
    if not hasattr(app.state, "estimator"):
        raise HTTPException(503, "Model is not loaded yet.")

    rest_img, dorsi_img = await _decode(image_rest), await _decode(image_dorsi)
    # Inference is CPU/GPU-bound: keep it off the event loop.
    rest_angle, rest_b64 = await run_in_threadpool(_analyse, rest_img, "Resting image")
    dorsi_angle, dorsi_b64 = await run_in_threadpool(_analyse, dorsi_img, "Dorsiflexion image")

    return {
        "angle_rest": round(rest_angle, 2),
        "angle_dorsi": round(dorsi_angle, 2),
        "rom": round(abs(dorsi_angle - rest_angle), 2),
        "image_rest": rest_b64,    # base64 JPEG, no data-URI prefix
        "image_dorsi": dorsi_b64,
    }
