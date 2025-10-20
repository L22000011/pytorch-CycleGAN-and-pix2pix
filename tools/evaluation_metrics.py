import torch
import lpips  # pip install lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

# 初始化 LPIPS 网络
lpips_alex = lpips.LPIPS(net='alex').eval()  # 可选择 net='vgg' 或 'alex'

def compute_lpips(img1, img2):
    """
    img1, img2: torch.Tensor, shape [1,3,H,W], range [-1,1]
    返回 LPIPS 感知距离
    """
    with torch.no_grad():
        dist = lpips_alex(img1, img2)
    return dist.item()

def compute_ssim(img1, img2):
    """
    img1, img2: numpy [H,W,C], range 0-255
    返回平均 SSIM
    """
    ssim_vals = []
    for i in range(img1.shape[2]):
        ssim_vals.append(ssim(img1[:,:,i], img2[:,:,i], data_range=255))
    return np.mean(ssim_vals)

def compute_psnr(img1, img2):
    """
    img1, img2: numpy [H,W,C], range 0-255
    返回平均 PSNR
    """
    psnr_vals = []
    for i in range(img1.shape[2]):
        psnr_vals.append(psnr(img1[:,:,i], img2[:,:,i], data_range=255))
    return np.mean(psnr_vals)

def compute_quality_score(img1, img2):
    """
    综合指标，越高表示质量越好
    """
    s = compute_ssim(img1, img2)
    p = compute_psnr(img1, img2)
    return 0.5 * s + 0.5 * (p / 50)  # PSNR/50 归一化到 0~1
