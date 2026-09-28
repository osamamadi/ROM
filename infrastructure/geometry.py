"""Pure math + drawing helpers (no model dependencies).

Image coordinates throughout: x to the right, y DOWNWARD.
Keypoint array layout (load-bearing, matches the v2 model): 0 bottom_heel, 1 5th_metatarsal, 2 malleolus.
"""
import base64
import math

import cv2
import numpy as np

Point = tuple[float, float]

HEEL, MT5, MALL = 0, 1, 2
KEYPOINT_NAMES = ["bottom_heel", "5th_metatarsal", "malleolus"]

CROP_PAD_PX = 200        # minimum padding around the points' bounding box
CROP_PAD_FRAC = 0.35     # ...or this fraction of the box's longest side, whichever is larger
OUT_MIN_SIDE = 900       # upscale small crops so annotations stay crisp
OUT_MAX_SIDE = 1280      # ...and cap large ones (payload size + memory)

# BGR
C_AXIS, C_VERT, C_ARC = (255, 170, 0), (255, 255, 255), (0, 200, 255)
C_HEEL, C_META, C_MALL, C_STAR = (60, 60, 255), (80, 210, 60), (0, 190, 255), (255, 80, 200)


# ------------------------------------------------------------------ math
def construct_heel(k: np.ndarray) -> np.ndarray:
    """Heel placed directly BELOW the malleolus, at the sole level."""
    return np.array([k[MALL, 0], k[HEEL, 1]], dtype=float)


def ankle_angle(k: np.ndarray) -> float:
    """Included angle (deg) between heel*->5th_metatarsal and the image vertical (up).
    Neutral foot ~ 90 deg; dorsiflexion makes it smaller."""
    v = k[MT5] - construct_heel(k)
    bearing = math.degrees(math.atan2(v[1], v[0]))
    d = (bearing - (-90.0) + 180.0) % 360.0 - 180.0
    return abs(d)


def rom(k_rest: np.ndarray, k_dorsi: np.ndarray) -> float:
    return abs(ankle_angle(k_dorsi) - ankle_angle(k_rest))


# --------------------------------------------------------------- drawing
def crop_to_points(img: np.ndarray, pts: dict[str, Point]) -> tuple[np.ndarray, dict[str, Point]]:
    """Crop to the points' padded bounding box (clamped to the image); return crop + shifted points."""
    h, w = img.shape[:2]
    xs, ys = [p[0] for p in pts.values()], [p[1] for p in pts.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = max(CROP_PAD_PX, CROP_PAD_FRAC * max(x1 - x0, y1 - y0))

    cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    cx1, cy1 = min(w, int(math.ceil(x1 + pad))), min(h, int(math.ceil(y1 + pad)))
    return img[cy0:cy1, cx0:cx1], {n: (x - cx0, y - cy0) for n, (x, y) in pts.items()}


def _fit_size(img: np.ndarray, pts: dict[str, Point]) -> tuple[np.ndarray, dict[str, Point]]:
    """Scale so the longest side lies within [OUT_MIN_SIDE, OUT_MAX_SIDE]; scale points to match."""
    longest = max(img.shape[:2])
    s = OUT_MIN_SIDE / longest if longest < OUT_MIN_SIDE else min(1.0, OUT_MAX_SIDE / longest)
    if abs(s - 1.0) < 1e-3:
        return img, pts
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=s, fy=s, interpolation=interp), {n: (x * s, y * s) for n, (x, y) in pts.items()}


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    """Text on a dark plate so it stays legible on any background."""
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    pad = max(4, int(th * 0.4))
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_angle_arc(img: np.ndarray, vertex: Point, a: Point, b: Point, angle: float, scale: float) -> None:
    """Semi-transparent pie + outline at `vertex` spanning the included angle between rays to a and b."""
    va, vb = np.subtract(a, vertex), np.subtract(b, vertex)
    radius = int(max(30, min(np.linalg.norm(va), np.linalg.norm(vb)) * 0.55))
    center = (int(vertex[0]), int(vertex[1]))

    a1 = math.degrees(math.atan2(va[1], va[0]))
    a2 = math.degrees(math.atan2(vb[1], vb[0]))
    sweep = (a2 - a1 + 180) % 360 - 180          # signed shortest sweep in (-180, 180]
    start = a1 if sweep >= 0 else a2
    sweep = abs(sweep)                            # cv2.ellipse angles run clockwise in image coords

    overlay = img.copy()
    cv2.ellipse(overlay, center, (radius, radius), 0, start, start + sweep, C_ARC, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
    cv2.ellipse(img, center, (radius, radius), 0, start, start + sweep, C_ARC, max(2, round(3 * scale)), cv2.LINE_AA)

    mid = math.radians(start + sweep / 2)
    lx = int(center[0] + math.cos(mid) * (radius + 28 * scale))
    ly = int(center[1] + math.sin(mid) * (radius + 28 * scale))
    text = f"{angle:.1f} deg"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.95 * scale, max(2, round(2 * scale)))
    lx = int(np.clip(lx - tw / 2, 8, img.shape[1] - tw - 8))
    ly = int(np.clip(ly + th / 2, th + 8, img.shape[0] - 12))
    _label(img, text, (lx, ly), 0.95 * scale)


def draw_overlay(img_bgr: np.ndarray, k: np.ndarray, angle: float) -> np.ndarray:
    """Annotate the (EXIF-corrected) image and crop it to the ankle region. `k` is (3, 2) in image pixels.

    Draws: the 3 predicted points, the constructed heel* (star), the vertical through the malleolus
    (from heel* upward), the foot axis heel*->5th_metatarsal and the angle between them.
    """
    heel_star = construct_heel(k)
    up_len = max(abs(k[MALL, 1] - heel_star[1]), 0.5 * np.linalg.norm(k[MT5] - heel_star), 50.0)
    vert_top = (heel_star[0], heel_star[1] - up_len * 1.15)

    pts: dict[str, Point] = {
        "heel": tuple(k[HEEL]), "mt5": tuple(k[MT5]), "mall": tuple(k[MALL]),
        "star": tuple(heel_star), "top": vert_top,
    }
    crop, p = crop_to_points(img_bgr, pts)
    out, p = _fit_size(crop.copy(), p)
    scale = max(out.shape[:2]) / 900
    ip = {n: (int(round(x)), int(round(y))) for n, (x, y) in p.items()}

    limb_t, radius, ring = max(4, round(6 * scale)), max(9, round(14 * scale)), max(2, round(3 * scale))

    _draw_angle_arc(out, p["star"], p["top"], p["mt5"], angle, scale)

    # Vertical reference (dashed look via two passes) and foot axis, with a dark outline for contrast.
    for a, b, col in (("star", "top", C_VERT), ("star", "mt5", C_AXIS)):
        cv2.line(out, ip[a], ip[b], (20, 20, 20), limb_t + 4, cv2.LINE_AA)
        cv2.line(out, ip[a], ip[b], col, limb_t, cv2.LINE_AA)

    for name, key, color in (("HEEL", "heel", C_HEEL), ("5TH MT", "mt5", C_META), ("MALLEOLUS", "mall", C_MALL)):
        pt = ip[key]
        cv2.circle(out, pt, radius + ring, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, radius, color, -1, cv2.LINE_AA)
        fs = 0.7 * scale
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, round(2 * scale)))
        # HEEL label goes to the left: heel* usually sits right next to it, on the right.
        tx = pt[0] - radius - 12 - tw if key == "heel" else pt[0] + radius + 12
        tx = int(np.clip(tx, 6, out.shape[1] - tw - 14))
        ty = int(np.clip(pt[1] - radius - 8, th + 10, out.shape[0] - 14))
        _label(out, name, (tx, ty), fs)

    # heel*: star marker, drawn last so it stays visible even when it lands on top of the heel point
    cv2.drawMarker(out, ip["star"], (255, 255, 255), cv2.MARKER_STAR, int(radius * 3.4), ring + 3, cv2.LINE_AA)
    cv2.drawMarker(out, ip["star"], C_STAR, cv2.MARKER_STAR, int(radius * 3.4), ring, cv2.LINE_AA)

    sx, sy = ip["star"]
    _label(out, "heel*", (int(np.clip(sx + radius * 3, 6, out.shape[1] - 120)),
                          int(np.clip(sy + radius * 4, 30, out.shape[0] - 14))), 0.7 * scale)
    return out


def to_base64_jpeg(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")
