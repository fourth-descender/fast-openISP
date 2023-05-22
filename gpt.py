import numpy as np

data = np.arange(64).reshape((8, 8))
print(data)

# forwards = np.array([
#     [1, 1], [3, 3]
# ])

# forwards = np.array([
#     [1, 1], [3, 3]
# ])
# backwards = np.array([
#     [1, 3], [3, 1]
# ])

# data_copy = np.pad(data, (0, 1), mode='constant')

# t = np.where(forwards[:, 0] >= 1, data_copy[forwards[:, 0] + 1, forwards[:, 1] - 1], 0)

# print()
def interpolate_q(source):
    f_indices = np.array([[1, 1], [3, 3]])
    b_indices = np.array([[1, 3], [3, 1]])
    # padding to solve index out of bound issue with np.where
    source_copy = np.pad(source, (0, 1), mode='constant')
    f_l_cond = (f_indices[:, 1] >= 1) & (f_indices[:, 0] < source.shape[0] - 2)
    f_r_cond = (f_indices[:, 0] >= 1) & (f_indices[:, 1] < source.shape[1] - 2)
    b_l_cond = (b_indices[:, 0] >= 1) & (b_indices[:, 1] < source.shape[1] - 2)
    b_r_cond = (b_indices[:, 1] < source.shape[1] - 2) & (b_indices[:, 0] < source.shape[1] - 2)
    
    f_l = np.where(f_l_cond, source_copy[f_indices[:, 0] + 1, f_indices[:, 1] - 1], 0)
    f_r = np.where(f_r_cond, source_copy[f_indices[:, 0] - 1, f_indices[:, 0] + 1], 0)
    b_l = np.where(b_l_cond, source_copy[b_indices[:, 0] - 1, b_indices[:, 1] - 1], 0)
    b_r = np.where(b_r_cond, source_copy[b_indices[:, 0] + 1, b_indices[:, 1] + 1], 0)

    f_sum = f_l + f_r
    f_count = f_l_cond.astype(int) + f_r_cond.astype(int)

    b_sum = b_l + b_r
    b_count = b_l_cond.astype(int) + b_r_cond.astype(int)

    template = np.zeros(source.shape)
    template.put(np.ravel_multi_index(f_indices.T, source.shape), f_sum // f_count)
    template.put(np.ravel_multi_index(b_indices.T, source.shape), b_sum // b_count)
    print(template)
    # total = t + l  + d  + r
    # count = t_cond.astype(int) + l_cond.astype(int) + d_cond.astype(int) + r_cond.astype(int)
    # source.put(np.ravel_multi_index(indices.T, source.shape), total // count)

interpolate_q(data)