# tools/gradcam_logger.py

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def log_gradcam(model, input_tensor, layer_name='layer4', save_dir='gradcam_results', filename='gradcam.png', epoch=None, iteration=None):
    """
    对输入张量生成 Grad-CAM，并保存到 save_dir
    Args:
        model: PyTorch 模型
        input_tensor: 输入图片张量 (1, C, H, W)
        layer_name: 用于 Grad-CAM 的卷积层名
        save_dir: 保存 Grad-CAM 图片的根目录
        filename: 保存文件名
        epoch: 当前训练轮数（可选，用于目录区分）
        iteration: 当前训练迭代（可选，用于目录区分）
    """
    # 处理目录，如果 epoch 或 iteration 是数字，把它转成字符串
    if epoch is not None:
        save_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    if iteration is not None:
        save_dir = os.path.join(save_dir, f"iter_{iteration:06d}")
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 1. 获取目标卷积层
    target_layer = dict([*model.named_modules()])[layer_name]

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # 2. 前向
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # 3. 反向传播得到梯度
    model.zero_grad()
    output[0, pred_class].backward()

    # 4. 计算 Grad-CAM
    gradient = gradients[0].detach()[0]  # (C, H, W)
    activation = activations[0].detach()[0]  # (C, H, W)
    weights = gradient.mean(dim=(1, 2))  # 全局平均池化
    cam = (weights[:, None, None] * activation).sum(dim=0)  # (H, W)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # 5. 可视化叠加到原图
    img = input_tensor[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 6. 移除 hook
    forward_handle.remove()
    backward_handle.remove()

    print(f"[GradCAM] 已保存: {save_path}")
    return save_path


# 示例调用（测试用，可注释掉）
if __name__ == "__main__":
    model = resnet50(pretrained=True)
    dummy_img = torch.rand(1, 3, 224, 224)
    log_gradcam(model, dummy_img, epoch=1, iteration=8)
