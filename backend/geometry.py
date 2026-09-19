"""Pure math + drawing helpers (no model dependencies)."""
import base64

import cv2
import numpy as np

Point = tuple[float, float]


def included_angle(vertex: Point, a: Point, b: Point) -> float:
    """Angle in degrees at `vertex` between rays vertex->a and vertex->b."""
    va = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    vb = np.asarray(b, dtype=float) - np.asarray(vertex, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        raise ValueError("Degenerate keypoints: two points coincide.")
    cos = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def draw_skeleton(img_bgr: np.ndarray, kpts: dict[str, Point], angle: float) -> np.ndarray:
    """Return a copy of the image with keypoints, limbs and the angle drawn on it."""
    out = img_bgr.copy()
    scale = max(out.shape[:2]) / 800  # keep annotations readable at any resolution
    thick, radius = max(2, round(3 * scale)), max(4, round(7 * scale))

    heel = tuple(int(v) for v in kpts["bottom_heel"])
    meta = tuple(int(v) for v in kpts["5th_metatarsal"])
    shin = tuple(int(v) for v in kpts["shin"])

    cv2.line(out, heel, meta, (255, 170, 0), thick, cv2.LINE_AA)
    cv2.line(out, heel, shin, (255, 170, 0), thick, cv2.LINE_AA)
    for name, pt, color in (
        ("heel", heel, (0, 0, 255)),
        ("5th MT", meta, (0, 200, 0)),
        ("shin", shin, (0, 200, 255)),
    ):
        cv2.circle(out, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, pt, radius, (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
        cv2.putText(out, name, (pt[0] + radius + 4, pt[1] - radius), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)

    cv2.putText(out, f"{angle:.1f} deg", (heel[0] + 3 * radius, heel[1] + 4 * radius),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 170, 0), thick, cv2.LINE_AA)
    return out


def to_base64_jpeg(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")
