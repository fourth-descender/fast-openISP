import math
import numpy as np
from .basic_module import BasicModule

# makes sure numpy prints integers.
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
        self.r_const = self.params.r_const
        self.g_const = self.params.g_const
        self.b_const = self.params.b_const

    def execute(self, data):
        source = data['bayer'].astype(np.int32)
        # gets only the IR without the zeroes. 
        q_source = source[1::2, 1::2]
        # perform nearest neighbour upsampling to fill the lonely 1 in 4 IRs.
        q_filled = self.upsample(q_source, 2, 2)
        # transforms IRs to red, then reds to blue.
        self.interpolate(source)
        # subtract each 2x2 BGGR block with the corresponding IR value.
        self.minus_q(source, q_filled, self.r_const, self.g_const, self.b_const)
        # stores RGB and IR values for future use.
        data['bayer'] = source.astype(np.uint16)
        data['grayscale'] = q_filled


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
    
    def index(self, color):
        return np.column_stack(np.where(self.truth_table(self.get_base_table(color))))
    
    def interpolate_red(self, source):
        # the goal is to replace each red value with the nonzero average of the 
        # surrounding blues two units away.
        indices = self.index('R')
        # padding to solve index out of bound issue with np.where
        source_copy = np.pad(source, (0, 2), mode='constant')

        # creates the boolean filter for each possible blue positions (top, left, down, and right).
        t_cond = indices[:, 0] >= 2
        l_cond = indices[:, 1] >= 2
        d_cond = indices[:, 0] < source.shape[0] - 2
        r_cond = indices[:, 1] < source.shape[1] - 2

        # apply the filter and get the values.
        t = np.where(t_cond, source_copy[indices[:, 0] - 2, indices[:, 1]], 0)
        l = np.where(l_cond, source_copy[indices[:, 0], indices[:, 1] - 2], 0)
        d = np.where(d_cond, source_copy[indices[:, 0] + 2, indices[:, 1]], 0)
        r = np.where(r_cond, source_copy[indices[:, 0], indices[:, 1] + 2], 0)

        # get the nonzero average.
        total = t + l  + d  + r
        count = t_cond.astype(int) + l_cond.astype(int) + d_cond.astype(int) + r_cond.astype(int)

        # updates source.
        source.put(np.ravel_multi_index(indices.T, source.shape), total // count)
    
    def interpolate_q(self, source):
        # the main idea is to split into two parts: forward and backward
        # forward refers to the IR indices that accepts top right and bottom left red.
        # backward refers to the IR indices that accepts bottom right and top left.
        f_indices = self.index('f')
        b_indices = self.index('b')

        # padding to solve index out of bound issue with np.where
        source_copy = np.pad(source, (0, 1), mode='constant')

        # create boolean filters for each red positions that may surround an IR.
        # in other words, forward_left (bottom left), forward_right (top right),
        # backward_left (top left), and backward_right (bottom_right).
        f_l_cond = (f_indices[:, 0] >= 1) & (f_indices[:, 1] < source.shape[1] - 1)
        f_r_cond = (f_indices[:, 0] < source.shape[0] - 1) & (f_indices[:, 1] >= 1)
        b_l_cond = (b_indices[:, 0] >= 1) & (b_indices[:, 1] >= 1)
        b_r_cond = (b_indices[:, 0] < source.shape[0] - 1) & (b_indices[:, 1] < source.shape[1] - 1)

        # apply the filter and get those values.
        f_l = np.where(f_l_cond, source_copy[f_indices[:, 0] - 1, f_indices[:, 1] + 1], 0)
        f_r = np.where(f_r_cond, source_copy[f_indices[:, 0] + 1, f_indices[:, 1] - 1], 0)
        b_l = np.where(b_l_cond, source_copy[b_indices[:, 0] - 1, b_indices[:, 1] - 1], 0)
        b_r = np.where(b_r_cond, source_copy[b_indices[:, 0] + 1, b_indices[:, 1] + 1], 0)

        # interpolates values for forward and backward IRs.
        f_sum = f_l + f_r
        f_count = f_l_cond.astype(int) + f_r_cond.astype(int)

        # bottom right IR (forward interpolation) has no reds to use.
        f_count[f_count == 0] = 1
        f = f_sum // f_count
        f[f == 0] = source[-1, -1]

        b_sum = b_l + b_r
        b_count = b_l_cond.astype(int) + b_r_cond.astype(int)

        #  updates source.
        source.put(np.ravel_multi_index(f_indices.T, source.shape), f)
        source.put(np.ravel_multi_index(b_indices.T, source.shape), b_sum // b_count)

    def interpolate(self, source):
        # IR -> R first, then R -> B.
        self.interpolate_q(source)
        self.interpolate_red(source)

    def minus_q(self, source, q_source, r_const, g_const, b_const):
        # create boolean masks for each respective color
        r_filter = self.new_truth_table(self.get_new_base_table("n_r"))
        g_filter = self.new_truth_table(self.get_new_base_table("n_g"))
        b_filter = self.new_truth_table(self.get_new_base_table("n_b"))

        # multiply the upsampled IR blocks with the color constants.
        r_q = r_const * q_source
        g_q = g_const * q_source
        b_q = b_const * q_source

        # keep only for when the boolean filter is true.
        r = np.where(r_filter, r_q.astype(np.uint32), 0)
        g = np.where(g_filter, g_q.astype(np.uint32), 0)
        b = np.where(b_filter, b_q.astype(np.uint32), 0)

        # update actual source (python updates list/array parameters).
        source -= (r + g + b)
