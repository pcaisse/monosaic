import unittest
import monosaic
import numpy as np
from PIL import Image

class RGBColorTests(unittest.TestCase):

    def test_validation(self):
        orange = monosaic.RGBColor(255, 165, 0)
        orange2 = monosaic.RGBColor([255, 165, 0])
        self.assertTrue(orange == [255, 165, 0])
        self.assertTrue(orange == orange2)
        with self.assertRaises(ValueError):
            monosaic.RGBColor([256, 165, 0])


class HelperTests(unittest.TestCase):

    def test_name_from_path(self):
        self.assertTrue(monosaic.name_from_path('my_file.txt') == 'my_file')
        self.assertTrue(monosaic.name_from_path('my_file.tar.gz') == 'my_file')

    def test_euclid_distance(self):
        a = np.array([1, 2, 3])
        b = np.array([2, 2, 3])
        self.assertTrue(monosaic.euclid_distance(a, b) == 1.0)


class ImageTests(unittest.TestCase):

    def setUp(self):
        self.island_img = Image.open('test_imgs/island.jpeg')

    def test_average_image_color(self):
        avg_island_img_color = monosaic.average_image_color(self.island_img)
        self.assertTrue(np.array_equal(avg_island_img_color, np.array([55, 123, 162])))

if __name__ == '__main__':
    unittest.main()
