"""Dev check: the slim checkpoint must give IDENTICAL keypoints to the full one.

    python tools/verify_slim.py /path/to/epoch_160.pth [image.jpg]

Uses a synthetic image if none is given. Does not delete the full checkpoint.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imaging import prepare_image  # noqa: E402
from memory import log_rss  # noqa: E402
from pose import PoseEstimator, make_slim_checkpoint, slim_path  # noqa: E402

CONFIG = os.getenv("MODEL_CONFIG", "configs/hrnet_w32_ankle_v2.py")


def main() -> None:
    full = sys.argv[1]
    if len(sys.argv) > 2:
        data = open(sys.argv[2], "rb").read()
    else:  # deterministic synthetic 3:4 portrait image
        import io
        from PIL import Image
        rng = np.random.default_rng(0)
        arr = rng.integers(60, 200, (2048, 1536, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=90)
        data = buf.getvalue()
    img = prepare_image(data).bgr

    log_rss("before full load")
    k_full, s_full = PoseEstimator(CONFIG, full).predict(img)
    log_rss("after full-model predict")

    slim = make_slim_checkpoint(full, delete_full=False) if not os.path.exists(slim_path(full)) else slim_path(full)
    k_slim, s_slim = PoseEstimator(CONFIG, slim).predict(img)

    dk, ds = np.abs(k_full - k_slim).max(), np.abs(s_full - s_slim).max()
    print(f"full : {os.path.getsize(full) / 2**20:.0f} MB   slim: {os.path.getsize(slim) / 2**20:.0f} MB")
    print(f"keypoints full={k_full.round(3).tolist()}")
    print(f"max |keypoint diff| = {dk}   max |score diff| = {ds}")
    print("IDENTICAL" if dk == 0 and ds == 0 else "MISMATCH")
    sys.exit(0 if dk == 0 and ds == 0 else 1)


if __name__ == "__main__":
    main()
