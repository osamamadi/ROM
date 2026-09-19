"""Pure math + drawing helpers (no model dependencies)."""
import base64
import math

import cv2
import numpy as np

Point = tuple[float, float]

CROP_PAD_PX = 200        # minimum padding around the keypoints' bounding box
CROP_PAD_FRAC = 0.35     # ...or this fraction of the box's longest side, whichever is larger
OUT_MIN_SIDE = 900       # upscale small crops so annotations stay crisp
OUT_MAX_SIDE = 1600      # ...and cap huge ones to keep the JSON payload reasonable

# BGR
C_LIMB = (255, 170, 0)
C_ARC = (0, 200, 255)
C_HEEL, C_META, C_SHIN = (60, 60, 255), (80, 210, 60), (0, 190, 255)


def included_angle(vertex: Point, a: Point, b: Point) -> float:
    """Angle in degrees at `vertex` between rays vertex->a and vertex->b."""
    va = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    vb = np.asarray(b, dtype=float) - np.asarray(vertex, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        raise ValueError("Degenerate keypoints: two points coincide.")
    cos = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def crop_to_keypoints(img: np.ndarray, kpts: dict[str, Point]) -> tuple[np.ndarray, dict[str, Point]]:
    """Crop to the keypoints' padded bounding box (clamped to the image); return crop + shifted keypoints."""
    h, w = img.shape[:2]
    xs, ys = [p[0] for p in kpts.values()], [p[1] for p in kpts.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = max(CROP_PAD_PX, CROP_PAD_FRAC * max(x1 - x0, y1 - y0))

    cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    cx1, cy1 = min(w, int(math.ceil(x1 + pad))), min(h, int(math.ceil(y1 + pad)))
    crop = img[cy0:cy1, cx0:cx1]
    return crop, {k: (x - cx0, y - cy0) for k, (x, y) in kpts.items()}


def _fit_size(img: np.ndarray, kpts: dict[str, Point]) -> tuple[np.ndarray, dict[str, Point]]:
    """Scale the crop so its longest side is within [OUT_MIN_SIDE, OUT_MAX_SIDE]; scale keypoints to match."""
    longest = max(img.shape[:2])
    s = OUT_MIN_SIDE / longest if longest < OUT_MIN_SIDE else min(1.0, OUT_MAX_SIDE / longest)
    if abs(s - 1.0) < 1e-3:
        return img, kpts
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
    return out, {k: (x * s, y * s) for k, (x, y) in kpts.items()}


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    """Text on a dark rounded-ish plate so it stays legible on any background."""
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    pad = max(4, int(th * 0.4))
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_angle_arc(img: np.ndarray, heel: Point, a: Point, b: Point, angle: float, scale: float) -> None:
    """Semi-transparent filled pie + outline at `heel` spanning the included angle between rays to a and b."""
    va, vb = np.subtract(a, heel), np.subtract(b, heel)
    radius = int(max(30, min(np.linalg.norm(va), np.linalg.norm(vb)) * 0.55))
    center = (int(heel[0]), int(heel[1]))

    a1 = math.degrees(math.atan2(va[1], va[0]))
    a2 = math.degrees(math.atan2(vb[1], vb[0]))
    sweep = (a2 - a1 + 180) % 360 - 180          # signed shortest sweep, in (-180, 180]
    start = a1 if sweep >= 0 else a2
    sweep = abs(sweep)                            # cv2.ellipse angles run clockwise in image coords

    overlay = img.copy()
    cv2.ellipse(overlay, center, (radius, radius), 0, start, start + sweep, C_ARC, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
    cv2.ellipse(img, center, (radius, radius), 0, start, start + sweep, C_ARC, max(2, round(3 * scale)), cv2.LINE_AA)

    # Angle value on the bisector, just outside the arc.
    mid = math.radians(start + sweep / 2)
    lx = int(center[0] + math.cos(mid) * (radius + 28 * scale))
    ly = int(center[1] + math.sin(mid) * (radius + 28 * scale))
    text = f"{angle:.1f} deg"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.95 * scale, max(2, round(2 * scale)))
    lx = int(np.clip(lx - tw / 2, 8, img.shape[1] - tw - 8))
    ly = int(np.clip(ly + th / 2, th + 8, img.shape[0] - 12))
    _label(img, text, (lx, ly), 0.95 * scale, (255, 255, 255))


def draw_skeleton(img_bgr: np.ndarray, kpts: dict[str, Point], angle: float) -> np.ndarray:
    """Crop to the ankle region, then draw limbs, keypoints and the angle arc. Returns a new image."""
    crop, k = crop_to_keypoints(img_bgr, kpts)
    out, k = _fit_size(crop.copy(), k)
    scale = max(out.shape[:2]) / 900

    limb_t = max(4, round(6 * scale))
    radius = max(9, round(14 * scale))
    ring = max(2, round(3 * scale))

    heel, meta, shin = k["bottom_heel"], k["5th_metatarsal"], k["shin"]
    hp, mp, sp = (tuple(int(round(v)) for v in p) for p in (heel, meta, shin))

    _draw_angle_arc(out, heel, meta, shin, angle, scale)

    # Limbs: dark outline underneath for contrast on any skin/background tone.
    for p in (mp, sp):
        cv2.line(out, hp, p, (20, 20, 20), limb_t + 4, cv2.LINE_AA)
    for p in (mp, sp):
        cv2.line(out, hp, p, C_LIMB, limb_t, cv2.LINE_AA)

    for name, pt, color in (("HEEL", hp, C_HEEL), ("5TH MT", mp, C_META), ("SHIN", sp, C_SHIN)):
        cv2.circle(out, pt, radius + ring, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, radius, color, -1, cv2.LINE_AA)
        fs = 0.7 * scale
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, round(2 * scale)))
        tx = int(np.clip(pt[0] + radius + 12, 6, out.shape[1] - tw - 14))
        ty = int(np.clip(pt[1] - radius - 8, th + 10, out.shape[0] - 14))
        _label(out, name, (tx, ty), fs)
    return out


def to_base64_jpeg(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")
