"""
Create new mosaic image using one source image and a target image.
"""

from __future__ import print_function
from PIL import Image
import numpy as np
from operator import itemgetter
import argparse
import os
import datetime
import time


DEFAULT_FILE_FORMAT = 'jpg'
DEFAULT_NUM_COLOR_GROUPS = 64
DEFAULT_TILE_SIZE = 5
MAX_NUM_COLOR_GROUPS = 256
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class RGBColor(list):
    """
    Custom list with three RGB values (0-255 inclusive).
    eg) [255, 0, 255]
    """
    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) in [list, tuple]:
            args = args[0]
        if len(args) != 3 or not all([RGBColor._is_valid_value(arg) for arg in args]):
            raise ValueError('Invalid RGB color values')
        super(RGBColor, self).__init__(args)
    @staticmethod
    def _is_valid_value(value):
        return type(value) is int and 0 <= value <= 255


def name_from_path(path):
    """
    Returns just the name of the file (without file extension).
    """
    return os.path.basename(path).split('.')[0]


def euclid_distance(x, y):
    """
    Calculate the euclidean distance.
    See: http://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    """
    return np.sqrt(np.sum((x - y) ** 2))


def channel_avg_value(color_channel):
    """
    Get weighted average of each color channel (R, G, or B).
    """
    channel_sum = sum(color_channel)
    if channel_sum == 0:
        return 0
    # the *index* is the channel value, and the *value* is its weight
    return sum(value * weight for value, weight in enumerate(color_channel)) / channel_sum


def average_image_color(img):
    """
    Find average image color within image.
    See: https://gist.github.com/olooney/1246268
    """
    histogram = img.histogram()

    red = histogram[0:256]
    green = histogram[256:256*2]
    blue = histogram[256*2: 256*3]

    return np.array(
        RGBColor(
            channel_avg_value(red),
            channel_avg_value(green),
            channel_avg_value(blue),
        )
    )


def img_reduced_palette_colors(img, num_color_groups):
    """
    Returns a reduced color palette best representing that image.
    NB: This reduced color palette is used for optimization. When searching for the source tile
        that best matches the target, we only want to consider a small subset of similar tiles,
        not all of them. Without this optimization, complexity is O(n^2), whereas with it it's O(n).
    """
    imgp = img.convert("P", palette=Image.ADAPTIVE, colors=num_color_groups)
    return [RGBColor(color[1]) for color in imgp.convert("RGB").getcolors()]


def get_tile_img(img, tile_index, num_tiles_per_row, tile_size):
    """
    Get tile image from larger image.
    """
    row_num = tile_index / num_tiles_per_row
    col_num = tile_index % num_tiles_per_row
    x = col_num * tile_size
    y = row_num * tile_size
    box = (x, y, x + tile_size, y + tile_size)
    return box, img.crop(box)


def get_curr_tile_index(curr_row_index, curr_col_index, num_tiles_per_row):
    """
    Get current tile index. 
    """
    return curr_row_index * num_tiles_per_row + curr_col_index


def get_avg_colors(img, tile_data, tile_size):
    """
    Find average colors for each tile in image.
    """
    avg_colors = []
    for curr_row in xrange(tile_data['num_rows']):
        for curr_col in xrange(tile_data['num_tiles_per_row']):
            tile_index = get_curr_tile_index(curr_row, curr_col, tile_data['num_tiles_per_row'])
            box, img_tile = get_tile_img(img, tile_index, tile_data['num_tiles_per_row'], tile_size)
            avg_color = average_image_color(img_tile)
            avg_colors.append((tile_index, avg_color))
    return avg_colors


def get_matching_tile_index(avg_target_tile_color, source_color_groups, reduced_palette_colors):
    """
    Get index of source image tile that most closely matches target tile color.
    """
    sorted_distances = get_sorted_color_matches(avg_target_tile_color, reduced_palette_colors)
    target_color_group_indexes = get_best_color_match_indexes(sorted_distances)
    # Sometimes the best match for the target color isn't the best match for any source color
    # In that case, find the next closest match that matches a source color group
    comparison_colors = [source_color_groups[index] if index in source_color_groups else get_next_best_color_match_index(sorted_distances, source_color_groups) for index in target_color_group_indexes][0]
    distances = calc_distances(avg_target_tile_color, comparison_colors)
    min_result = min(distances, key=itemgetter(1))
    return min_result[0]


def get_color_groups(colors, reduced_palette_colors):
    """
    Find best color matches for all colors.
    """
    color_groups = {}
    for color in colors:
        sorted_distances = get_sorted_color_matches(color[1], reduced_palette_colors)
        indexes = get_best_color_match_indexes(sorted_distances)
        for index in indexes:
            if index not in color_groups:
                color_groups[index] = []
            color_groups[index].append(color)
    return color_groups


def calc_distances(single_color, iterable):
    """
    Find distance from a single color for every color in iterable.
    """
    return [(index, euclid_distance(color, single_color)) for index, color in iterable]


def get_best_color_match_indexes(sorted_distances):
    """
    Determine which color(s) out of a reduced palette this color most closely matches.
    """
    # use all colors that are sufficiently close in smallest distance
    best_match = sorted_distances[0]
    best_matches = filter(lambda c: int(c[1]) == int(best_match[1]), sorted_distances)
    return zip(*best_matches)[0]


def get_next_best_color_match_index(sorted_distances, source_color_groups):
    """
    Find the next best match that is in a source color group.
    """
    for distance in sorted_distances:
        index = distance[0]
        if index in source_color_groups:
            return source_color_groups[index]


def get_sorted_color_matches(single_color, reduced_palette_colors):
    """
    Return reduced palette colors sorted by closest match to single color.
    """
    distances = calc_distances(single_color, enumerate(reduced_palette_colors))
    return sorted(distances, key=itemgetter(1))


def get_img_tile_data(img_width, img_height, tile_size):
    """
    Get tile data for image based on tile size.
    """
    num_rows = img_height / tile_size
    num_tiles_per_row = img_width / tile_size
    num_tiles = num_rows * num_tiles_per_row
    return {
        'num_rows': num_rows,
        'num_tiles_per_row': num_tiles_per_row,
        'num_tiles': num_tiles
    }

def cropped_img_dimensions(img_width, img_height, tile_size):
    """
    Return image dimensions cropped to a size that is a multiple of tile size.
    """
    img_width = img_width - (img_width % tile_size)
    img_height = img_height - (img_height % tile_size)
    return img_width, img_height


def create_img(source_img_path, target_img_path, tile_size=None, output_dir=None, file_format=None, color_groups=None):
    """
    Create a new image using a source image and target image.

    Source image = tiles used to create new image
    Target image = what we want the new image to look like

    The new image is generated by iterating over tiles in the target image and finding the best matching
    source tile to use for the new image. The best match is determined by cross-comparing the average colors
    of the source and target image tiles. A reduced color palette is used to limit the number of comparisons
    to other similarly colored tiles for performance reasons.
    """
    source_img = Image.open(source_img_path)
    target_img = Image.open(target_img_path)

    source_img_width, source_img_height = source_img.size
    target_img_width, target_img_height = target_img.size

    tile_size = int(tile_size) if tile_size else DEFAULT_TILE_SIZE
    color_groups = min(int(color_groups), MAX_NUM_COLOR_GROUPS) if color_groups else DEFAULT_NUM_COLOR_GROUPS

    target_img_width, target_img_height = cropped_img_dimensions(target_img_width, target_img_height, tile_size)

    source_img_tile_data = get_img_tile_data(source_img_width, source_img_height, tile_size)
    target_img_tile_data = get_img_tile_data(target_img_width, target_img_height, tile_size)

    print("Analyzing colors...")
    source_avg_colors = get_avg_colors(source_img, source_img_tile_data, tile_size)
    reduced_palette_colors = img_reduced_palette_colors(source_img, color_groups)

    source_color_groups = get_color_groups(source_avg_colors, reduced_palette_colors)

    new_im = Image.new('RGB', (target_img_width, target_img_height))

    tile_cache = {}

    print("Building new image...")
    for curr_row in xrange(target_img_tile_data['num_rows']):
        for curr_col in xrange(target_img_tile_data['num_tiles_per_row']):
            target_tile_index = get_curr_tile_index(curr_row, curr_col, target_img_tile_data['num_tiles_per_row'])

            target_tile_box, target_img_tile = get_tile_img(target_img, target_tile_index, target_img_tile_data['num_tiles_per_row'], tile_size)
            avg_target_tile_color = average_image_color(target_img_tile)
            avg_target_tile_color_key = '-'.join(map(str, avg_target_tile_color))

            if avg_target_tile_color_key in tile_cache:
                # Another tile had the same average color, so used cached value
                source_tile_index = tile_cache[avg_target_tile_color_key]
            else:
                # Find index of best matching source tile
                source_tile_index = get_matching_tile_index(avg_target_tile_color, source_color_groups, reduced_palette_colors)
                tile_cache[avg_target_tile_color_key] = source_tile_index

            source_tile_box, source_tile_img = get_tile_img(source_img, source_tile_index, source_img_tile_data['num_tiles_per_row'], tile_size)

            new_im.paste(source_tile_img, target_tile_box)
            percent_done = ((target_tile_index + 1) / float(target_img_tile_data['num_tiles'])) * 100
            print('{percent_done:.1f}% done'.format(percent_done=percent_done), end='\r')

    filename = name_from_path(source_img_path) + '_' + name_from_path(target_img_path) + '_' + \
               datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + '.' + (file_format or DEFAULT_FILE_FORMAT)
    output_dir = os.path.join(output_dir or SCRIPT_DIR, filename)

    new_im.save(output_dir)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--source', help='Source image (tiles used to create new image)', required=True)
    parser.add_argument('-t', '--target', help='Target image (what to make new image look like)', required=True)
    parser.add_argument('-z', '--tile-size', help='Optional size for tiles (default is {tile_size} pixels)'.format(tile_size=DEFAULT_TILE_SIZE))
    parser.add_argument('-o', '--output-dir', help='Optional directory path for output file (default is directory where file is located)')
    parser.add_argument('-f', '--file-format', help='File format (default is {format})'.format(format=DEFAULT_FILE_FORMAT))
    parser.add_argument('-g', '--color-groups', 
        help='Number of color groups used for determining best color match (default is {default}, max is {max})'.format(
            default=DEFAULT_NUM_COLOR_GROUPS,
            max=MAX_NUM_COLOR_GROUPS,
        )
    )

    args = parser.parse_args()

    start_time = time.time()

    path_to_new_img = create_img(args.source, args.target, args.tile_size, args.output_dir, args.file_format, args.color_groups)

    time_delta = time.time() - start_time
    print('Image created in {sec:.1f} seconds. Saved at: {path}'.format(sec=time_delta, path=path_to_new_img))


if __name__ == '__main__':
    main()
