import os
import numpy as np
from tools.evaluation_metrics import compute_lpips, compute_quality_score
import torch
from PIL import Image
from torchvision import transforms

# 图片预处理
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # [-1,1]
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def quality_filter(fake_path, ref_path, threshold=0.5):
    """
    筛选低质量图片
    fake_path: 生成图像路径
    ref_path: 对应参考图像路径
    threshold: 分数低于阈值则认为质量差
    返回: True=质量好, False=质量差
    """
    fake_img = load_image(fake_path)
    ref_img = load_image(ref_path)

    # 转为 numpy [H,W,C], 0-255
    fake_np = np.array(fake_img).astype(np.float32)
    ref_np = np.array(ref_img).astype(np.float32)

    score = compute_quality_score(fake_np, ref_np)
    return score >= threshold

def lpips_score(fake_path, ref_path):
    """
    使用 LPIPS 计算感知距离
    """
    fake_img = to_tensor(load_image(fake_path)).unsqueeze(0)  # [1,3,H,W]
    ref_img = to_tensor(load_image(ref_path)).unsqueeze(0)
    return compute_lpips(fake_img, ref_img)
