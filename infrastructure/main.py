"""Ankle ROM API: stateless FastAPI service around an MMPose HRNet-W32 model."""
import gc
import logging
import os

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from geometry import KEYPOINT_NAMES, ankle_angle, draw_overlay, rom, to_base64_jpeg
from imaging import prepare_image
from memory import log_rss
from pose import PoseEstimator, ensure_model_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rom.api")


def parse_origins(raw: str) -> list[str]:
    """'a, https://x.app/ ,b' -> ['a', 'https://x.app', 'b'] (whitespace and trailing slashes removed)."""
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


# --- Configuration (override via env vars on Railway) -------------------------------------
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?id=1WVB3HL6lhEHg8uQCbAmC_Ed8lOAtHalx")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/weights/epoch_160.pth")
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "configs/hrnet_w32_ankle_v2.py")  # MMPose training config (v2)
DEVICE = os.getenv("DEVICE", "cpu")
RAW_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = parse_origins(RAW_ALLOWED_ORIGINS)
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
LOW_CONF = 0.5

app = FastAPI(title="Ankle ROM API")


# Registered BEFORE CORSMiddleware so it sits INSIDE it: an unhandled error becomes a JSON 500 that still
# gets CORS headers (Starlette's own 500 handler sits outside CORS, so the browser would show "CORS error").
@app.middleware("http")
async def json_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def load_model() -> None:
    """Download weights (once), slim them, and load the model into memory. Runs at boot, never per request."""
    log.info("ALLOWED_ORIGINS raw env value: %r -> parsed: %s", RAW_ALLOWED_ORIGINS, ALLOWED_ORIGINS)
    log.info("Env vars whose name mentions ORIGIN/CORS: %s", sorted(k for k in os.environ if "ORIGIN" in k.upper() or "CORS" in k.upper()))
    torch.set_num_threads(TORCH_THREADS)
    log_rss("startup (before model)")
    checkpoint = ensure_model_file(MODEL_URL, MODEL_PATH)
    log_rss("after download/slim")
    app.state.estimator = PoseEstimator(MODEL_CONFIG, checkpoint, device=DEVICE)
    gc.collect()
    log_rss("after model load")
    log.info("Model ready.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": hasattr(app.state, "estimator")}


def _check_size(upload: UploadFile) -> None:
    """413 (JSON, so CORS headers are attached) before any bytes are pulled into memory."""
    if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename}: file is {upload.size / 2**20:.0f} MB, the limit is "
                                 f"{MAX_UPLOAD_BYTES // 2**20} MB.")


def _analyse(holder: list[bytes], label: str) -> tuple[dict, list[str], object]:
    """One image -> (result dict, warnings, keypoints in original-image pixels).

    `holder` carries the raw upload bytes so they can be released as soon as the array is decoded.
    """
    try:
        img = prepare_image(holder.pop())  # no reference to the raw bytes survives this call
        log_rss(f"{label}: decoded")
        kpts_small, scores = app.state.estimator.predict(img.bgr)
    except ValueError as e:
        raise HTTPException(422, f"{label}: {e}")
    log_rss(f"{label}: inference done")

    kpts = kpts_small / img.scale  # back to the (EXIF-corrected) original image size
    angle = ankle_angle(kpts)
    warnings = [
        f"{label}: low confidence for {n} ({s:.2f})" for n, s in zip(KEYPOINT_NAMES, scores) if s < LOW_CONF
    ]
    result = {
        "angle_deg": round(angle, 2),
        "keypoints": {n: [round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(KEYPOINT_NAMES, kpts)},
        "scores": {n: round(float(s), 3) for n, s in zip(KEYPOINT_NAMES, scores)},
        # overlay is drawn on the downscaled image (never the full-res decode), long side <= 1280
        "overlay": to_base64_jpeg(draw_overlay(img.bgr, kpts_small, angle)),  # base64 JPEG, no data-URI prefix
    }
    del img
    gc.collect()
    log_rss(f"{label}: overlay drawn")
    return result, warnings, kpts


@app.post("/api/calculate_rom")
async def calculate_rom(image_rest: UploadFile = File(...), image_dorsi: UploadFile = File(...)) -> dict:
    if not hasattr(app.state, "estimator"):
        raise HTTPException(503, "Model is not loaded yet.")

    log_rss("request start")
    _check_size(image_rest)
    _check_size(image_dorsi)

    # Strictly sequential, one upload in memory at a time; raw bytes are dropped as soon as they are decoded.
    results = []
    for upload, label in ((image_rest, "Resting image"), (image_dorsi, "Dorsiflexion image")):
        holder = [await upload.read()]
        size_mb = len(holder[0]) / 2**20
        log_rss(f"{label}: upload read ({size_mb:.1f} MB)")
        if size_mb * 2**20 > MAX_UPLOAD_BYTES:  # fallback when the server did not report upload.size
            raise HTTPException(413, f"{upload.filename}: file is {size_mb:.0f} MB, the limit is {MAX_UPLOAD_BYTES // 2**20} MB.")
        # Inference is CPU-bound: keep it off the event loop.
        results.append(await run_in_threadpool(_analyse, holder, label))
        gc.collect()
    (rest, w1, k_rest), (dorsi, w2, k_dorsi) = results

    return {"rom_deg": round(rom(k_rest, k_dorsi), 2), "rest": rest, "dorsi": dorsi, "warnings": w1 + w2}
