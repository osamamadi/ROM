"""Ankle ROM API: stateless FastAPI service around an MMPose HRNet-W32 model."""
import logging
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from geometry import KEYPOINT_NAMES, ankle_angle, draw_overlay, rom, to_base64_jpeg
from imaging import prepare_image
from pose import PoseEstimator, ensure_weights

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rom.api")

# --- Configuration (override via env vars on Railway) -------------------------------------
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?id=1WVB3HL6lhEHg8uQCbAmC_Ed8lOAtHalx")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/weights/epoch_160.pth")
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "configs/hrnet_w32_ankle_v2.py")  # MMPose training config (v2)
DEVICE = os.getenv("DEVICE", "cpu")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOW_CONF = 0.5

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


async def _read(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename}: file exceeds 10 MB.")
    return data


def _analyse(data: bytes, label: str) -> tuple[dict, list[str]]:
    """One image -> result dict (keypoints in ORIGINAL pixel coords, scores, angle, overlay) + warnings."""
    try:
        img = prepare_image(data)
        kpts_small, scores = app.state.estimator.predict(img.small_bgr)
    except ValueError as e:
        raise HTTPException(422, f"{label}: {e}")

    kpts = kpts_small / img.scale  # back to the (EXIF-corrected) original image size
    angle = ankle_angle(kpts)
    warnings = [
        f"{label}: low confidence for {n} ({s:.2f})" for n, s in zip(KEYPOINT_NAMES, scores) if s < LOW_CONF
    ]
    result = {
        "angle_deg": round(angle, 2),
        "keypoints": {n: [round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(KEYPOINT_NAMES, kpts)},
        "scores": {n: round(float(s), 3) for n, s in zip(KEYPOINT_NAMES, scores)},
        "overlay": to_base64_jpeg(draw_overlay(img.full_bgr, kpts, angle)),  # base64 JPEG, no data-URI prefix
        "_kpts": kpts,
    }
    return result, warnings


@app.post("/api/calculate_rom")
async def calculate_rom(image_rest: UploadFile = File(...), image_dorsi: UploadFile = File(...)) -> dict:
    if not hasattr(app.state, "estimator"):
        raise HTTPException(503, "Model is not loaded yet.")

    rest_bytes, dorsi_bytes = await _read(image_rest), await _read(image_dorsi)
    # Inference is CPU/GPU-bound: keep it off the event loop.
    rest, w1 = await run_in_threadpool(_analyse, rest_bytes, "Resting image")
    dorsi, w2 = await run_in_threadpool(_analyse, dorsi_bytes, "Dorsiflexion image")

    rom_deg = rom(rest.pop("_kpts"), dorsi.pop("_kpts"))
    return {"rom_deg": round(rom_deg, 2), "rest": rest, "dorsi": dorsi, "warnings": w1 + w2}
