import os
import time
import math
import numpy as np
from .basic_module import BasicModule

np.set_printoptions(
    suppress=True, 
    formatter={
        'int_kind':'{:d}'.format
    }
)

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

class FGP(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.RESOLUTION = (self.cfg.hardware.raw_height, self.cfg.hardware.raw_width)
        self.KERNEL_SIZE = (4, 4)
        self.KERNEL = {
            "R": r_indices,
            "G": g_indices,
            "B": b_indices,
            "Q": q_indices,
            "f": forward,
            "b": backward
        }

    def execute(self, data):
        source = data['bayer'].astype(np.int32)
        self.interpolate(source)
        data['bayer'] = source.astype(np.uint16)

    def get_base_table(self, color):
        indices = self.KERNEL[color]
        color_filter = np.zeros(self.KERNEL_SIZE, dtype=bool)
        color_filter[indices] = True
        return color_filter
    
    def truth_table(self, block):
        v_repetitions = math.floor(self.RESOLUTION[0] / self.KERNEL_SIZE[0])
        h_repetitions = math.floor(self.RESOLUTION[1] / self.KERNEL_SIZE[1])
        truth_table = np.tile(block, (v_repetitions, h_repetitions))
        return truth_table
    
    def filter(self, source, color):
        boolean_filter = self.truth_table(self.get_base_table(color))
        return np.where(boolean_filter, source, 0)
    
    def index(self, color):
        return np.column_stack(np.where(self.truth_table(self.get_base_table(color))))
    
    def index_separate(self, color):
        return np.where(self.truth_table(self.get_base_table(color)))
    
    def interpolate_red(self, source):
        indices = self.index('R')
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
    
    def interpolate_q(self, source):
        # f_row, f_col = np.where(truth_table(get_base_table('f')))
        # b_row, b_col = np.where(truth_table(get_base_table('b')))
        f_row, f_col = self.index_separate('f')
        b_row, b_col = self.index_separate('b')
    
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
    
    
    def interpolate(self, source):
        self.interpolate_q(source)
        self.interpolate_red(source)
        

# OUTPUT_DIR = './output'
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# RESOLUTION = (1944, 2592)
# file_location = 'raw/sample.raw'
# raw_data = np.fromfile(
#             file_location, 
#             dtype='uint16', 
#             sep='').reshape(RESOLUTION)




# def save_to_txt(name, obj):
#     output_path = os.path.join(OUTPUT_DIR, name)
#     np.savetxt(output_path, obj, fmt='%d')

# start = time.time()
# interpolate(raw_data)
# save_to_txt('fuck_you_interpolation.txt', raw_data)
# end = time.time()
# print(end - start)
