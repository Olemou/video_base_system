import torch
import torch.nn as nn
def rotate_half(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)
