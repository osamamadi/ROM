"""Model download + MMPose inference wrapper."""
import logging
import os

import gdown
import numpy as np
from mmpose.apis import inference_topdown, init_model

from geometry import Point

log = logging.getLogger("rom.pose")

DEFAULT_KEYPOINT_ORDER = ["bottom_heel", "5th_metatarsal", "shin"]


def ensure_weights(url: str, dest: str) -> str:
    """Download the checkpoint from Google Drive once (skipped if already on disk)."""
    if os.path.exists(dest):
        log.info("Weights already present at %s", dest)
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    log.info("Downloading weights to %s ...", dest)
    if not gdown.download(url, dest, quiet=False, fuzzy=True):
        raise RuntimeError("gdown failed to download the model weights.")
    return dest


class PoseEstimator:
    def __init__(self, config: str, checkpoint: str, device: str = "cpu"):
        self.model = init_model(config, checkpoint, device=device)
        meta = getattr(self.model, "dataset_meta", None) or {}
        id2name = meta.get("keypoint_id2name")
        self.names = [id2name[i] for i in sorted(id2name)] if id2name else DEFAULT_KEYPOINT_ORDER
        missing = set(DEFAULT_KEYPOINT_ORDER) - set(self.names)
        if missing:
            raise RuntimeError(f"Model is missing expected keypoints: {sorted(missing)}")

    def predict(self, img_bgr: np.ndarray) -> dict[str, Point]:
        # bboxes=None -> the whole image is used as the bbox (foot photo assumed to fill frame).
        results = inference_topdown(self.model, img_bgr)
        if not results:
            raise ValueError("No pose detected.")
        kpts = results[0].pred_instances.keypoints[0]  # (K, 2)
        return {name: (float(x), float(y)) for name, (x, y) in zip(self.names, kpts)}
