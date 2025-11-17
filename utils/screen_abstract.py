import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_gaussian_kernel(size, sigma):
    """生成一个高斯核"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()

# 定义 16x16 的高斯核
gaussian_kernel = torch.tensor(create_gaussian_kernel(16, sigma=4), dtype=torch.float32)

def gamearea_abstract(gamescreen):
    reshaped = gamescreen.unfold(0, 16, 16).unfold(1, 16, 16)
    pooled = (reshaped * gaussian_kernel).sum(dim=(-1, -2))
    pooled_int8 = pooled.clamp(0, 255).to(torch.uint8)  # 限制范围在 0-255，并转换为 uint8
    return pooled_int8