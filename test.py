import os
import os.path as op

import cv2
import time
import math
import numpy as np
import skimage.io

from pipeline import Pipeline
from utils.yacs import Config

np.set_printoptions(
    suppress=True, 
    formatter={
        'int_kind':'{:d}'.format
    }
)

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

KERNEL_SIZE = (4, 4)
RESOLUTION = (1944, 2592)
file_location = 'raw/sample.raw'
raw_data = np.fromfile(
            file_location, 
            dtype='uint16', 
            sep='').reshape(RESOLUTION)

r_indices = (
    (0, 2), (2, 0)
)
        
g_indices = (
    (0, 1), (0, 3), 
    (1, 0), (1, 2),
    (2, 1), (2, 3),
    (3, 0), (3, 2)
)

b_indices = (
    (0, 0), (2, 2)
)

q_indices = (
    (1, 1), (1, 3),
    (3, 1), (3, 3)
)

color_kernel = {
    "R": r_indices,
    "G": g_indices,
    "B": b_indices,
    "Q": q_indices
}

def get_base_table(color):
    indices = color_kernel[color]
    kernel_truth_table = np.zeros(KERNEL_SIZE, dtype=bool)
    for row, col in indices:
        kernel_truth_table[row][col] = True
    return kernel_truth_table


def truth_table(source, block):
    v_repetitions = math.floor(RESOLUTION[0] / KERNEL_SIZE[0])
    h_repetitions = math.floor(RESOLUTION[1] / KERNEL_SIZE[1])
    truth_table = np.tile(block, (v_repetitions, h_repetitions))
    return truth_table

# reds = filter_by_color(raw_data, "R")
# greens = filter_by_color(raw_data, "G")
# blues = filter_by_color(raw_data, "B")
# infrareds = filter_by_color(raw_data, "Q")
# # print(infrareds)

# def filter_rows_by_color(source, color):
#     return np.asarray(
#         [source[row] if color[row].any() else [0] * RESOLUTION[1] 
#          for row in range(RESOLUTION[0])]
#         )

# def filter_columns_by_color(source, color):
#     color_transpose = color.transpose()
#     source_transpose = source.transpose()
#     return np.asarray(
#         [source_transpose[column] if color_transpose[column].any() else [0] * RESOLUTION[0] 
#          for column in range(RESOLUTION[1])]
#         ).transpose()

def get_color_index(source, color):
    return np.column_stack(np.where(truth_table(source, get_base_table(color))))

def build(source, indices):
    template = np.zeros(source.shape, dtype="uint16")
    for index in indices:
        template[index[0]][index[1]] = source[index[0]][index[1]]
    return template
    
def filter_by_color(source, color):
    return build(source, get_color_index(source, color))

# red_indices = get_color_index(raw_data, "R")
# print(red_indices)
# print(build(raw_data, red_indices))

print(filter_by_color(raw_data, "R"))
print(filter_by_color(raw_data, "G"))
print(filter_by_color(raw_data, "B"))
print(filter_by_color(raw_data, "Q"))
# mod_green = filter_rows_by_color(greens, reds)
# print(reds)
# print(greens)
# print(mod_green)
