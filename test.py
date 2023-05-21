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
    # rows, cols
    (0, 2), (2, 0)
)

g_indices = (
    (0, 0, 1, 1, 2, 2, 3, 3),
    (1, 3, 0, 2, 1, 3, 0, 2)
)

b_indices = (
    (0, 2), (0, 2)
)

q_indices = (
    (1, 1, 3, 3), (1, 3, 1, 3)
)

forward = (
    (1, 3), (1, 3)
)

backward = (
    (1, 3), (3, 1)
)

KERNEL = {
    "R": r_indices,
    "G": g_indices,
    "B": b_indices,
    "Q": q_indices,
    "f": forward,
    "b": backward
}

def get_base_table(color):
    indices = KERNEL[color]
    color_filter = np.zeros(KERNEL_SIZE, dtype=bool)
    color_filter[indices] = True
    return color_filter

def truth_table(block):
    v_repetitions = math.floor(RESOLUTION[0] / KERNEL_SIZE[0])
    h_repetitions = math.floor(RESOLUTION[1] / KERNEL_SIZE[1])
    truth_table = np.tile(block, (v_repetitions, h_repetitions))
    return truth_table

def filter(source, color):
    boolean_filter = truth_table(get_base_table(color))
    return np.where(boolean_filter, source, 0)

def index(color):
    return np.column_stack(np.where(truth_table(get_base_table(color))))

def index_separate(color):
    return np.where(truth_table(get_base_table(color)))

def interpolate_red(source):
    indices = index('R')
    # padding to solve index out of bound issue with np.where
    source_copy = np.pad(source, (0, 2), mode='constant')
    t_cond = indices[:, 0] >= 2
    l_cond = indices[:, 1] >= 2
    d_cond = indices[:, 0] < source.shape[0] - 2
    r_cond = indices[:, 1] < source.shape[1] - 2
    t = np.where(t_cond, source_copy[indices[:, 0] - 2, indices[:, 1]], 0)
    l = np.where(l_cond, source_copy[indices[:, 0], indices[:, 1] - 2], 0)
    d = np.where(d_cond, source_copy[indices[:, 0] + 2, indices[:, 1]], 0)
    r = np.where(r_cond, source_copy[indices[:, 0], indices[:, 1] + 2], 0)
    total = t + l  + d  + r
    count = t_cond.astype(int) + l_cond.astype(int) + d_cond.astype(int) + r_cond.astype(int)
    source.put(np.ravel_multi_index(indices.T, source.shape), total // count)

def interpolate_q(source):
    # f_row, f_col = np.where(truth_table(get_base_table('f')))
    # b_row, b_col = np.where(truth_table(get_base_table('b')))
    f_row, f_col = index_separate('f')
    b_row, b_col = index_separate('b')

    f_l = np.zeros(f_row.shape)
    f_r = np.zeros(f_col.shape)

    b_l = np.zeros(b_row.shape)
    b_r = np.zeros(b_col.shape)

    f_l_valid = (f_row - 1 >= 0) & (f_col + 1 < source.shape[1])
    f_r_valid = (f_row + 1 < source.shape[0]) & (f_col - 1 >= 0)

    b_l_valid = (b_row - 1 >= 0) & (b_col - 1 >= 0)
    b_r_valid = (b_row + 1 < source.shape[0]) & (b_col + 1 < source.shape[1])

    f_l[f_l_valid] = source[f_row[f_l_valid] - 1, f_col[f_l_valid] + 1]
    f_r[f_r_valid] = source[f_row[f_r_valid] + 1, f_col[f_r_valid] - 1]

    b_l[b_l_valid] = source[b_row[b_l_valid] - 1, b_col[b_l_valid] - 1]
    b_r[b_r_valid] = source[b_row[b_r_valid] + 1, b_col[b_r_valid] + 1]

    # edge case when there's no reds (don't know how to incorporate it to np.where)
    f_sum = f_l + f_r
    f_count = (f_l != 0).astype(int) + (f_r != 0).astype(int)
    f_count[f_count == 0] = 1
    f = f_sum // f_count
    f[f == 0] = source[-1, -1]

    b_sum = b_l + b_r
    b_count = (b_l != 0).astype(int) + (b_r != 0).astype(int)
    b_count[b_count == 0] = 1
    b = b_sum // b_count
    b[b == 0] = source[-1, -1]

    source[f_row, f_col] = f
    source[b_row, b_col] = b


def interpolate(source):
    interpolate_q(source)
    interpolate_red(source)

def save_to_txt(name, obj):
    output_path = os.path.join(OUTPUT_DIR, name)
    np.savetxt(output_path, obj, fmt='%d')

start = time.time()
# filter(raw_data, 'R')
# print(index('R'))
# interpolate_red(raw_data)
# print(interpolate_red(raw_data))
# print(interpolate_q(raw_data))
# print(raw_data)
# print(filter(raw_data, 'Q'))
# print(raw_data)
interpolate(raw_data)
save_to_txt('fuck_you_interpolation.txt', raw_data)

# print(get_base_table('f').astype(int))
# print(get_base_table('b').astype(int))
# print(get_base_table('R').astype(int))
# print(get_base_table('G').astype(int))
# print(get_base_table('B').astype(int))
# print(get_base_table('Q').astype(int))
# save_to_txt('source.txt', raw_data)
# save_to_txt('fuck_me.txt', interpolate_red(raw_data))
end = time.time()
print(end - start)
