"""Image decoding that mirrors the training preprocessing."""
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

MAX_LONG_SIDE = 2048


@dataclass
class PreparedImage:
    bgr: np.ndarray                 # EXIF-corrected, long side <= 2048; used for inference AND the overlay
    scale: float                    # bgr = original * scale
    orig_size: tuple[int, int]      # (width, height) of the EXIF-corrected original


def prepare_image(data: bytes) -> PreparedImage:
    """EXIF-transpose -> RGB -> LANCZOS downscale to long side <= 2048. Raises ValueError if not an image.

    The full-resolution decode is released before returning, so only the downscaled array stays alive.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # PIL raises many types for corrupt / unsupported data
        raise ValueError("not a valid image") from e

    w, h = img.size
    scale = min(1.0, MAX_LONG_SIDE / max(w, h))
    if scale != 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    # MMPose (like mmcv.imread) expects BGR arrays.
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    img.close()
    return PreparedImage(bgr=bgr, scale=scale, orig_size=(w, h))
