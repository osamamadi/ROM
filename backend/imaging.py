"""Image decoding that mirrors the training preprocessing."""
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

MAX_LONG_SIDE = 2048


@dataclass
class PreparedImage:
    full_bgr: np.ndarray    # EXIF-corrected, original resolution (used for the overlay)
    small_bgr: np.ndarray   # downscaled to long side <= 2048 (used for inference)
    scale: float            # small = full * scale


def prepare_image(data: bytes) -> PreparedImage:
    """EXIF-transpose -> RGB -> LANCZOS downscale to long side <= 2048. Raises ValueError if not an image."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # PIL raises many types for corrupt / unsupported data
        raise ValueError("not a valid image") from e

    w, h = img.size
    scale = min(1.0, MAX_LONG_SIDE / max(w, h))
    small = img if scale == 1.0 else img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    # MMPose (like mmcv.imread) expects BGR arrays.
    to_bgr = lambda im: np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
    return PreparedImage(full_bgr=to_bgr(img), small_bgr=to_bgr(small), scale=scale)
