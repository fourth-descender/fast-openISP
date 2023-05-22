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

new_b_indices = (
    (0), (0)
)

new_g_indices = (
    (0, 1), (1, 0)
)

new_r_indices = (
    (1), (1)
)

class FGP(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.RESOLUTION = (self.cfg.hardware.raw_height, self.cfg.hardware.raw_width)
        self.KERNEL_SIZE = (4, 4)
        self.NEW_KERNEL_SIZE = (2, 2)
        self.KERNEL = {
            "R": r_indices,
            "G": g_indices,
            "B": b_indices,
            "Q": q_indices,
            "f": forward,
            "b": backward,
            "n_r": new_r_indices,
            "n_g": new_g_indices,
            "n_b": new_b_indices
        }

    def execute(self, data):
        source = data['bayer'].astype(np.int32)
        q_source = source[1::2, 1::2]
        q_filled = self.upsample(q_source, 2, 2)
        self.interpolate(source)
        self.minus_q(source, q_filled, 0.717, 0.22, 0.375)
        data['bayer'] = source.astype(np.uint16)

    def upsample(self, source, x, y):
        row, col = source.shape
        row_stride, col_stride = source.strides
        return np.lib.stride_tricks.as_strided(source, (row, x, col, y), 
                                    (row_stride, 0, col_stride, 0)).reshape(row * x, col * y)
    
    def get_new_base_table(self, color):
        indices = self.KERNEL[color]
        color_filter = np.zeros(self.NEW_KERNEL_SIZE, dtype=bool)
        color_filter[indices] = True
        return color_filter

    def get_base_table(self, color):
        indices = self.KERNEL[color]
        color_filter = np.zeros(self.KERNEL_SIZE, dtype=bool)
        color_filter[indices] = True
        return color_filter
    
    def new_truth_table(self, block):
        v_repetitions = math.floor(self.RESOLUTION[0] / self.NEW_KERNEL_SIZE[0])
        h_repetitions = math.floor(self.RESOLUTION[1] / self.NEW_KERNEL_SIZE[1])
        truth_table = np.tile(block, (v_repetitions, h_repetitions))
        return truth_table

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
        f_indices = self.index('f')
        b_indices = self.index('b')
        # # padding to solve index out of bound issue with np.where

        source_copy = np.pad(source, (0, 1), mode='constant')

        f_l_cond = (f_indices[:, 0] >= 1) & (f_indices[:, 1] < source.shape[1] - 1)
        f_r_cond = (f_indices[:, 0] < source.shape[0] - 1) & (f_indices[:, 1] >= 1)
        b_l_cond = (b_indices[:, 0] >= 1) & (b_indices[:, 1] >= 1)
        b_r_cond = (b_indices[:, 0] < source.shape[0] - 1) & (b_indices[:, 1] < source.shape[1] - 1)

        f_l = np.where(f_l_cond, source_copy[f_indices[:, 0] - 1, f_indices[:, 1] + 1], 0)
        f_r = np.where(f_r_cond, source_copy[f_indices[:, 0] + 1, f_indices[:, 1] - 1], 0)
        b_l = np.where(b_l_cond, source_copy[b_indices[:, 0] - 1, b_indices[:, 1] - 1], 0)
        b_r = np.where(b_r_cond, source_copy[b_indices[:, 0] + 1, b_indices[:, 1] + 1], 0)

        f_sum = f_l + f_r
        f_count = f_l_cond.astype(int) + f_r_cond.astype(int)

        # bottom right IR (forward interpolation) has no reds to use.
        f_count[f_count == 0] = 1
        f = f_sum // f_count
        f[f == 0] = source[-1, -1]

        b_sum = b_l + b_r
        b_count = b_l_cond.astype(int) + b_r_cond.astype(int)

        source.put(np.ravel_multi_index(f_indices.T, source.shape), f)
        source.put(np.ravel_multi_index(b_indices.T, source.shape), b_sum // b_count)

    def interpolate(self, source):
        self.interpolate_q(source)
        self.interpolate_red(source)

    def minus_q(self, source, q_source, r_const, g_const, b_const):
        r_filter = self.new_truth_table(self.get_new_base_table("n_r"))
        g_filter = self.new_truth_table(self.get_new_base_table("n_g"))
        b_filter = self.new_truth_table(self.get_new_base_table("n_b"))
        r_q = r_const * q_source
        g_q = g_const * q_source
        b_q = b_const * q_source
        r = np.where(r_filter, r_q.astype(np.uint32), 0)
        g = np.where(g_filter, g_q.astype(np.uint32), 0)
        b = np.where(b_filter, b_q.astype(np.uint32), 0)
        source -= (r + g + b)
