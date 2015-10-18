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
        c = np.array([1, 2, 3])
        d = np.array([1, 2, 3])
        self.assertTrue(monosaic.euclid_distance(c, d) == 0.0)


class ImageTests(unittest.TestCase):

    def setUp(self):
        self.island_img = Image.open('test_imgs/island.jpeg')  # 226x170

    def test_cropped_img_dimensions(self):
        img_width, img_height = self.island_img.size
        self.assertTrue(monosaic.cropped_img_dimensions(img_width=img_width, img_height=img_height, tile_size=5) == (225, 170))

    def test_average_image_color(self):
        avg_island_img_color = monosaic.average_image_color(self.island_img)
        self.assertTrue(np.array_equal(avg_island_img_color, np.array([55, 123, 162])))

    def test_img_reduced_palette_colors(self):
        reduced_palette_colors = monosaic.img_reduced_palette_colors(self.island_img, num_color_groups=8)
        island_img_reduced_palette = [
            [161, 195, 218],
            [115, 157, 181],
            [52, 109, 133],
            [31, 133, 180],
            [65, 152, 199],
            [6, 52, 91],
            [4, 78, 130],
            [3, 102, 159],
        ]
        self.assertTrue(reduced_palette_colors == island_img_reduced_palette)

    def test_get_tile_img(self):
        cropped_first_island_img_22x22 = self.island_img.crop((0, 0, 10, 10))
        box, first_island_tile = monosaic.get_tile_img(self.island_img, tile_index=0, num_tiles_per_row=22, tile_size=10)
        self.assertTrue(first_island_tile, cropped_first_island_img_22x22)

    def test_get_img_tile_data(self):
        img_data = monosaic.get_img_tile_data(img_width=220, img_height=170, tile_size=10)
        self.assertTrue(img_data['num_rows'] == 17)
        self.assertTrue(img_data['num_tiles_per_row'] == 22)
        self.assertTrue(img_data['num_tiles'] == 374)

if __name__ == '__main__':
    unittest.main()
