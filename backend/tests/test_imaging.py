import io
import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imaging import prepare_image  # noqa: E402


def jpeg(w: int, h: int, orientation: int | None = None) -> bytes:
    rng = np.random.default_rng(1)
    im = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90, exif=exif)
    return buf.getvalue()


def reference(data: bytes) -> np.ndarray:
    """The training-style pipeline: full decode -> exif_transpose -> RGB -> LANCZOS to long side <= 2048."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    s = min(1.0, 2048 / max(im.size))
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im)[:, :, ::-1]


class ImagingTest(unittest.TestCase):
    def test_rig_size_photo_is_identical_to_full_decode(self):
        # 2316x3088: draft cannot shrink it, so the result must equal the plain pipeline bit for bit.
        data = jpeg(2316, 3088)
        p = prepare_image(data)
        self.assertEqual(p.bgr.shape[:2], (2048, 1536))
        self.assertTrue(np.array_equal(p.bgr, reference(data)))
        self.assertEqual(p.orig_size, (2316, 3088))

    def test_exif_rotation_swaps_original_size(self):
        p = prepare_image(jpeg(3088, 2316, orientation=6))  # stored landscape, displayed portrait
        self.assertEqual(p.orig_size, (2316, 3088))
        self.assertEqual(p.bgr.shape[:2], (2048, 1536))
        self.assertAlmostEqual(p.scale, 2048 / 3088)

    def test_huge_jpeg_uses_draft_and_still_hits_target_size(self):
        p = prepare_image(jpeg(6000, 4000))
        self.assertEqual(p.orig_size, (6000, 4000))
        self.assertEqual(p.bgr.shape[:2], (1365, 2048))

    def test_small_image_not_upscaled(self):
        p = prepare_image(jpeg(800, 600))
        self.assertEqual((p.scale, p.bgr.shape[:2]), (1.0, (600, 800)))

    def test_garbage_raises_value_error(self):
        with self.assertRaises(ValueError):
            prepare_image(b"not an image")


if __name__ == "__main__":
    unittest.main(verbosity=2)
