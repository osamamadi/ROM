"""Model download + MMPose inference wrapper."""
import gc
import logging
import os

import gdown
import numpy as np
import torch
from mmengine.config import Config
from mmpose.apis import inference_topdown, init_model

from geometry import KEYPOINT_NAMES

log = logging.getLogger("rom.pose")

HERE = os.path.dirname(os.path.abspath(__file__))
METAINFO_PATH = os.path.join(HERE, "configs", "ankle_v2_metainfo.py")


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


def slim_path(full_path: str) -> str:
    root, ext = os.path.splitext(full_path)
    return f"{root}_slim{ext}"


def make_slim_checkpoint(full_path: str, delete_full: bool = True) -> str:
    """Strip optimizer state etc. from the training checkpoint: keep only {state_dict, meta}.

    `meta` holds dataset_meta (keypoint names), which init_model reads. The full file is deleted afterwards.
    """
    dest = slim_path(full_path)
    ck = torch.load(full_path, map_location="cpu")
    log.info("Full checkpoint keys: %s", sorted(ck.keys()))
    slim = {"state_dict": ck["state_dict"], "meta": ck.get("meta", {})}
    del ck
    gc.collect()
    tmp = dest + ".tmp"
    torch.save(slim, tmp)

    # Prove the file on disk holds bit-identical weights before the full checkpoint is deleted.
    back = torch.load(tmp, map_location="cpu")
    same = back["state_dict"].keys() == slim["state_dict"].keys() and all(
        torch.equal(back["state_dict"][k], v) for k, v in slim["state_dict"].items()
    )
    has_meta = "dataset_meta" in back.get("meta", {})
    del back
    if not same:
        os.remove(tmp)
        raise RuntimeError("Slim checkpoint does not match the original state_dict.")
    log.info("Slim state_dict verified bit-identical (%d tensors); dataset_meta kept: %s", len(slim["state_dict"]), has_meta)

    os.replace(tmp, dest)  # atomic: a crash mid-write never leaves a truncated *_slim.pth
    del slim
    gc.collect()
    log.info("Slim checkpoint: %.0f MB -> %.0f MB", os.path.getsize(full_path) / 2**20, os.path.getsize(dest) / 2**20)
    if delete_full:
        os.remove(full_path)
    return dest


def ensure_model_file(url: str, full_path: str) -> str:
    """Return the path of the slim checkpoint, downloading + slimming the full one only if needed."""
    dest = slim_path(full_path)
    if os.path.exists(dest):
        log.info("Slim weights already present at %s", dest)
        return dest
    full = ensure_weights(url, full_path)
    try:
        return make_slim_checkpoint(full)
    except Exception:
        # Never take the service down over an optimisation: fall back to the full checkpoint.
        log.exception("Slimming failed; falling back to the full checkpoint (higher memory use).")
        return full


def _repoint_metainfo(node, path: str) -> None:
    """Recursively rewrite every `metainfo=dict(from_file=...)` to the container-local metainfo file."""
    if isinstance(node, dict):
        mi = node.get("metainfo")
        if isinstance(mi, dict) and "from_file" in mi:
            mi["from_file"] = path
        for v in node.values():
            _repoint_metainfo(v, path)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _repoint_metainfo(v, path)


def load_config(config_path: str) -> Config:
    """Load the training config and neutralise its Colab-only paths (DATA_ROOT is never read at inference)."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(HERE, config_path)
    for p in (config_path, METAINFO_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}. Copy the v2 config and metainfo into backend/configs/.")
    cfg = Config.fromfile(config_path)
    _repoint_metainfo(cfg._cfg_dict, METAINFO_PATH)  # Config is not a dict itself; walk its ConfigDict
    cfg.load_from = None
    # Custom training-only metrics (e.g. AnkleAngleMetric) are not registered in this app.
    for key in ("val_evaluator", "test_evaluator"):
        cfg.pop(key, None)
    return cfg


class PoseEstimator:
    def __init__(self, config_path: str, checkpoint: str, device: str = "cpu"):
        self.model = init_model(load_config(config_path), checkpoint, device=device)
        meta = getattr(self.model, "dataset_meta", None) or {}
        id2name = meta.get("keypoint_id2name")
        if id2name:
            names = [id2name[i] for i in sorted(id2name)]
            if names != KEYPOINT_NAMES:  # order is load-bearing for the angle geometry
                raise RuntimeError(f"Unexpected keypoints {names}, expected {KEYPOINT_NAMES}")

    def predict(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full-frame bbox (no detector). Returns keypoints (3, 2) in this image's pixels and scores (3,)."""
        h, w = img_bgr.shape[:2]
        with torch.inference_mode():
            results = inference_topdown(
                self.model, img_bgr, bboxes=np.array([[0, 0, w, h]], dtype=np.float32), bbox_format="xyxy"
            )
        if not results:
            raise ValueError("No pose detected.")
        inst = results[0].pred_instances
        return np.asarray(inst.keypoints[0], dtype=float), np.asarray(inst.keypoint_scores[0], dtype=float)
