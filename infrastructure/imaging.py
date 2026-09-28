"""Image decoding that mirrors the training preprocessing."""
import gc
import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from memory import log_rss

MAX_LONG_SIDE = 2048


@dataclass
class PreparedImage:
    bgr: np.ndarray                 # EXIF-corrected, long side <= 2048; used for inference AND the overlay
    scale: float                    # bgr = original * scale
    orig_size: tuple[int, int]      # (width, height) of the EXIF-corrected original


def prepare_image(data: bytes) -> PreparedImage:
    """[JPEG draft] -> EXIF-transpose -> RGB -> LANCZOS downscale to long side <= 2048.

    Raises ValueError if the bytes are not an image. Only the downscaled array outlives the call.
    """
    try:
        img = Image.open(io.BytesIO(data))
        w0, h0 = img.size
        # EXIF orientations 5-8 swap width/height; this is the size of the corrected ORIGINAL.
        ow, oh = (h0, w0) if img.getexif().get(0x0112, 1) in (5, 6, 7, 8) else (w0, h0)
        scale = min(1.0, MAX_LONG_SIDE / max(ow, oh))
        if img.format == "JPEG" and scale < 1:
            # DCT-domain downscale (1/2, 1/4, 1/8): PIL only applies it when the result stays >= the target in
            # BOTH dimensions, so images whose long side is < 4096 px (e.g. the 2316x3088 rig photos or
            # 12 MP phone shots) are decoded untouched and stay identical to the training preprocessing.
            img.draft("RGB", (math.ceil(w0 * scale), math.ceil(h0 * scale)))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # PIL raises many types for corrupt / unsupported data
        raise ValueError("not a valid image") from e
    log_rss("decode: PIL image loaded")

    target = (round(ow * scale), round(oh * scale))
    if img.size != target:
        img = img.resize(target, Image.LANCZOS)

    # MMPose (like mmcv.imread) expects BGR arrays.
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    img.close()
    del img
    gc.collect()
    log_rss("decode: array ready")
    return PreparedImage(bgr=bgr, scale=scale, orig_size=(ow, oh))
