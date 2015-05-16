"""
Create new mosaic image using one source image and a model image.
"""

from __future__ import print_function
from PIL import Image
import numpy as np
from operator import itemgetter
import argparse
import os
import itertools
import datetime
import time
# TODO: Use LAB instead of RGB
# from skimage.color import rgb2lab, lab2rgb


DEFAULT_FILE_FORMAT = 'jpg'
DEFAULT_NUM_COLOR_GROUPS = 64
DEFAULT_TILE_SIZE = 5
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


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


def average_image_color(tile):
    """
    Find average image color within tile.
    See: https://gist.github.com/olooney/1246268
    """
    i = Image.fromarray(tile)
    h = i.histogram()
 
    # split into red, green, blue
    r = h[0:256]
    g = h[256:256*2]
    b = h[256*2: 256*3]
 
    # perform the weighted average of each channel:
    # the *index* is the channel value, and the *value* is its weight
    return np.array((
        sum(i * w for i, w in enumerate(r)) / sum(r),
        sum(i * w for i, w in enumerate(g)) / sum(g),
        sum(i * w for i, w in enumerate(b)) / sum(b),
    ))


def img_reduced_colors(img, num_color_groups):
    """
    Returns a subset of all image colors best representing that image.
    NB: This reduced color palette is used for optimization. When searching for the source tile
        that best matches the model, we only want to consider a small subset of similar tiles,
        not all of them. Without this optimization, complexity is O(n^2), whereas with it it's O(n).
    """
    imgp = Image.fromarray(img).convert("P", palette=Image.ADAPTIVE, colors=num_color_groups)
    return [list(color[1]) for color in imgp.convert("RGB").getcolors()]


def get_tile_img_array(img, row_index, col_index, tile_size):
    """
    Get tile array from image array.
    """
    return img[row_index:row_index + tile_size, col_index:col_index + tile_size]


def get_tile_img(img, tile_index, num_rows, num_tiles_per_row, tile_size):
    """
    Get tile image from larger image.
    """
    row_num = tile_index / num_rows
    col_num = tile_index % num_tiles_per_row
    x = col_num * tile_size
    y = row_num * tile_size
    tile_array = get_tile_img_array(img, y, x, tile_size)
    return Image.fromarray(tile_array)


def get_avg_colors(img, num_rows, num_tiles_per_row, tile_size):
    """
    Find average colors for each tile in image.
    """
    avg_colors = []
    for i in xrange(num_rows):
        for j in xrange(num_tiles_per_row):
            img_tile = get_tile_img_array(img, i * tile_size, j * tile_size, tile_size)
            index = i * num_tiles_per_row + j
            val = average_image_color(img_tile)
            avg_colors.append((index, val))
    return avg_colors


def get_tile_index(avg_model_tile_color, source_color_groups, reduced_palette_colors):
    """
    Get index of source image tile that most closely matches model tile color.
    """
    color_group_indexes = get_color_match_indexes(avg_model_tile_color, reduced_palette_colors)
    comparison_colors = [source_color_groups[index] for index in itertools.chain(color_group_indexes)][0]
    distances = [(index, euclid_distance(color, avg_model_tile_color)) for index, color in comparison_colors]
    min_result = min(distances, key=itemgetter(1))
    return min_result[0]


def get_color_groups(colors, reduced_palette_colors):
    """
    Find best color matches for all colors.
    """
    color_groups = {}

    # all color groups need an empty list by default
    # in case there are no best matches for that group
    for i in range(len(reduced_palette_colors)):
        color_groups[i] = []

    for color in colors:
        indexes = get_color_match_indexes(color[1], reduced_palette_colors)
        for index in indexes:
            color_groups[index].append(color)
    return color_groups


def get_color_match_indexes(single_color, reduced_palette_colors):
    """
    Determine which color(s) out of a reduced palette this color most closely matches.
    """
    distances = [(index, euclid_distance(color[1], single_color)) for index, color in enumerate(reduced_palette_colors)]
    sorted_distances = sorted(distances, key=itemgetter(1))
    best_match = sorted_distances[0]
    # use all colors that are sufficiently close in smallest distance
    best_matches = filter(lambda c: int(c[1]) == int(best_match[1]), sorted_distances)
    return zip(*best_matches)[0]


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


def create_img(source_img_path, model_img_path, tile_size=None, output_dir=None, file_format=None, color_groups=None):
    """
    Create a new image using a source image and model image.
    """
    source_img = Image.open(source_img_path)
    model_img = Image.open(model_img_path)

    source_img_width, source_img_height = source_img.size
    model_img_width, model_img_height = model_img.size

    source_img = np.array(source_img)
    model_img = np.array(model_img)

    tile_size = int(tile_size) if tile_size else DEFAULT_TILE_SIZE
    color_groups = int(color_groups) if color_groups else DEFAULT_NUM_COLOR_GROUPS

    # Crop model image to a size that is a multiple of tile size
    model_img_width = model_img_width - (model_img_width % tile_size)
    model_img_height = model_img_height - (model_img_height % tile_size)
    model_img = model_img[0:model_img_height, 0:model_img_width]

    source_img_tile_data = get_img_tile_data(source_img_width, source_img_height, tile_size)
    model_img_tile_data = get_img_tile_data(model_img_width, model_img_height, tile_size)

    print("Analyzing colors...")
    source_avg_colors = get_avg_colors(source_img, source_img_tile_data['num_rows'], source_img_tile_data['num_tiles_per_row'], tile_size)
    reduced_palette_colors = img_reduced_colors(source_img, color_groups)

    source_color_groups = get_color_groups(source_avg_colors, reduced_palette_colors)

    new_im = Image.new('RGB', (model_img_width, model_img_height))

    tile_cache = {}

    print("Building new image...")
    for i in xrange(model_img_tile_data['num_rows']):
        for j in xrange(model_img_tile_data['num_tiles_per_row']):
            model_img_tile = get_tile_img_array(model_img, i * tile_size, j * tile_size, tile_size)
            avg_model_tile_color = average_image_color(model_img_tile)
            avg_model_tile_color_key = '-'.join(map(str, avg_model_tile_color))

            if avg_model_tile_color_key in tile_cache:
                tile_index = tile_cache[avg_model_tile_color_key]
            else:
                tile_index = get_tile_index(avg_model_tile_color, source_color_groups, reduced_palette_colors)
                tile_cache[avg_model_tile_color_key] = tile_index

            tile_img = get_tile_img(source_img, tile_index, source_img_tile_data['num_rows'], source_img_tile_data['num_tiles_per_row'], tile_size)

            count = (model_img_tile_data['num_tiles_per_row'] * i) + j
            percent_done = (count / float(model_img_tile_data['num_tiles'])) * 100
            print('{percent_done:.1f}% done'.format(percent_done=percent_done), end='\r')
            new_im.paste(tile_img, (j * tile_size, i * tile_size))

    filename = name_from_path(source_img_path) + '_' + name_from_path(model_img_path) + '_' + \
               datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + '.' + (file_format or DEFAULT_FILE_FORMAT)
    output_dir = os.path.join(output_dir or SCRIPT_DIR, filename)

    new_im.save(output_dir)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--source', help='Source image (tiles used to create new image)', required=True)
    parser.add_argument('-m', '--model', help='Model image (what to make new image look like)', required=True)
    parser.add_argument('-t', '--tile-size', help='Optional size for tiles (default is {tile_size} pixels)'.format(tile_size=DEFAULT_TILE_SIZE))
    parser.add_argument('-o', '--output-dir', help='Optional directory path for output file (default is directory where file is located)')
    parser.add_argument('-f', '--file-format', help='File format (default is {format})'.format(format=DEFAULT_FILE_FORMAT))
    parser.add_argument('-g', '--color-groups', 
        help='Number of color groups used for determining best color match (default is {groups})'.format(groups=DEFAULT_NUM_COLOR_GROUPS)
    )

    args = parser.parse_args()

    start_time = time.time()

    path_to_new_img = create_img(args.source, args.model, args.tile_size, args.output_dir, args.file_format, args.color_groups)

    time_delta = time.time() - start_time
    print('Image created in {sec:.1f} seconds. Saved at: {path}'.format(sec=time_delta, path=path_to_new_img))


if __name__ == '__main__':
    main()
