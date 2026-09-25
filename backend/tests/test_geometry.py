"""Run from backend/:  python -m unittest discover -s tests -v   (also works under pytest)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry import ankle_angle, draw_overlay, rom  # noqa: E402

# Real annotated points (native pixels) from the training data: [heel, 5th_metatarsal, malleolus]
REST = np.array([[1361.5, 2423.8], [726.8, 2435.6], [1355.2, 2066.3]])   # IMG_0578
DORSI = np.array([[1455.7, 2341.3], [844.7, 2121.3], [1458.0, 1950.9]])  # IMG_0586


class GeometryTest(unittest.TestCase):
    def test_angles_and_rom(self):
        self.assertAlmostEqual(ankle_angle(REST), 91.07, delta=0.01)
        self.assertAlmostEqual(ankle_angle(DORSI), 70.27, delta=0.01)
        self.assertAlmostEqual(rom(REST, DORSI), 20.80, delta=0.02)

    def test_overlay_renders_and_is_cropped(self):
        img = np.full((3000, 2250, 3), 127, np.uint8)
        out = draw_overlay(img, REST, ankle_angle(REST))
        self.assertEqual(out.ndim, 3)
        self.assertLess(out.shape[0] * out.shape[1], img.shape[0] * img.shape[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
