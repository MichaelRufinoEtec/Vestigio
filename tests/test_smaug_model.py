"""Testes locais do pré-processamento Smaug; não baixam nem carregam os pesos."""
import unittest
from PIL import Image

from backend.smaug_model import CROP, make_views


class SmaugViewTests(unittest.TestCase):
    def test_landscape_produces_three_square_native_resolution_views(self):
        image = Image.new("RGB", (1200, 600), "white")
        views = make_views(image, crops=3)
        self.assertEqual(len(views), 3)
        self.assertTrue(all(view.size == (CROP, CROP) for view in views))

    def test_portrait_produces_three_square_views(self):
        image = Image.new("RGB", (600, 1200), "white")
        views = make_views(image, crops=3)
        self.assertEqual(len(views), 3)
        self.assertTrue(all(view.size == (CROP, CROP) for view in views))

    def test_small_image_is_upscaled_to_model_crop_size(self):
        image = Image.new("RGB", (100, 80), "white")
        views = make_views(image, crops=1)
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].size, (CROP, CROP))

    def test_invalid_crop_count_is_rejected(self):
        with self.assertRaises(ValueError):
            make_views(Image.new("RGB", (600, 600)), crops=2)


if __name__ == "__main__":
    unittest.main()
