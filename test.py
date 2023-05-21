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

KERNEL = {
    "R": r_indices,
    "G": g_indices,
    "B": b_indices,
    "Q": q_indices
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

def interpolate_red(source):
    indices = index('R')
    source_copy = np.pad(source, (0, 2), mode='constant')
    template = np.zeros(source.shape)
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
    template.put(np.ravel_multi_index(indices.T, source.shape), total // count)
    return template

def save_to_txt(name, obj):
    output_path = os.path.join(OUTPUT_DIR, name)
    np.savetxt(output_path, obj, fmt='%d')

start = time.time()
# filter(raw_data, 'R')
# print(index('R'))
interpolate_red(raw_data)
# print(interpolate_red(raw_data))
# save_to_txt('fuck_me.txt', interpolate_red(raw_data))
end = time.time()
print(end - start)

# # reds = filter_by_color(raw_data, "R")
# # greens = filter_by_color(raw_data, "G")
# # blues = filter_by_color(raw_data, "B")
# # infrareds = filter_by_color(raw_data, "Q")
# # # print(infrareds)

# # def filter_rows_by_color(source, color):
# #     return np.asarray(
# #         [source[row] if color[row].any() else [0] * RESOLUTION[1] 
# #          for row in range(RESOLUTION[0])]
# #         )

# # def filter_columns_by_color(source, color):
# #     color_transpose = color.transpose()
# #     source_transpose = source.transpose()
# #     return np.asarray(
# #         [source_transpose[column] if color_transpose[column].any() else [0] * RESOLUTION[0] 
# #          for column in range(RESOLUTION[1])]
# #         ).transpose()

# def get_color_index(source, color):
#     return np.column_stack(np.where(truth_table(get_base_table(color))))

# def build(source, indices):
#     template = np.zeros(source.shape, dtype="uint16")
#     for index in indices:
#         template[index[0]][index[1]] = source[index[0]][index[1]]
#     return template
    
# def filter_by_color(source, color):
#     return build(source, get_color_index(source, color))

# # red_indices = get_color_index(raw_data, "R")
# # print(red_indices)
# # print(build(raw_data, red_indices))

# # print(filter_by_color(raw_data, "R"))
# # print(filter_by_color(raw_data, "G"))
# # print(filter_by_color(raw_data, "B"))
# # print(filter_by_color(raw_data, "Q"))
# # mod_green = filter_rows_by_color(greens, reds)
# # print(reds)
# # print(greens)
# # print(mod_green)


# def interpolate_point(source, index):
#     c = 0

#     t = 0
#     l = 0
#     d = 0
#     r = 0

#     row, col = index
#     t_index = row - 2
#     l_index = col - 2
#     d_index = row + 2
#     r_index = col + 2

#     if t_index > -1:
#         t = source[t_index][col]
#         c += 1
#     if l_index > -1:
#         l = source[row][l_index]
#         c += 1
#     if d_index < source.shape[0]:
#         d = source[d_index][col]
#         c += 1
#     if r_index < source.shape[1]:
#         r = source[row][r_index]
#         c += 1

#     source[row][col] = math.floor((t + d + l + r) / c)

# def interpolate_red(source):
#     color_indices = get_color_index(source, "R")
#     source_copy = source.copy()
#     for index in color_indices:
#         interpolate_point(source_copy, index)
#     return source_copy

# 
# start = time.time()
# print(filter_by_color(raw_data, "R"))
# # filter_by_color(interpolate_red(raw_data), "R")
# # save_to_txt("lmao.txt", filter_by_color(interpolate_red(raw_data), "R"))
# end = time.time()
# print(end - start)