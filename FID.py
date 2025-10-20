# offline_metrics_fixed_v2.py
import os
import glob
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net='alex').to(device)

# base_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_roi\organized"
base_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_edge\organized"

# 新增：创建 metrics 保存文件夹
metrics_dir = os.path.join(base_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)
txt_path = os.path.join(metrics_dir, "metrics_results.txt")
png_path = os.path.join(metrics_dir, "metrics_summary.png")

categories = [
    ('fake_A', 'real_A'),
    ('rec_A', 'real_A'),
    ('fake_B', 'real_B'),
    ('rec_B', 'real_B')
]

results = {}

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    return img

def compute_metrics(img1, img2):
    # LPIPS
    t1 = torch.tensor(img1).permute(2,0,1).unsqueeze(0)*2-1
    t2 = torch.tensor(img2).permute(2,0,1).unsqueeze(0)*2-1
    t1, t2 = t1.to(device), t2.to(device)
    lp = loss_fn(t1, t2).item()
    
    # SSIM
    h, w, _ = img1.shape
    win_size = 7
    if min(h, w) < win_size:
        win_size = min(h, w) // 2 * 2 + 1  # 取小于最小边的奇数
    s = ssim(img1, img2, channel_axis=2, win_size=win_size, data_range=1.0)
    
    # PSNR
    p = psnr(img1, img2, data_range=1.0)
    
    return lp, s, p

# ==========================
# 计算指标
# ==========================
for gen, real in categories:
    gen_dir = os.path.join(base_dir, gen)
    real_dir = os.path.join(base_dir, real)
    gen_imgs = sorted(glob.glob(os.path.join(gen_dir, "*.*")))
    real_imgs = sorted(glob.glob(os.path.join(real_dir, "*.*")))
    
    lp_list, s_list, p_list = [], [], []
    
    for g_path, r_path in zip(gen_imgs, real_imgs):
        g_img = load_image(g_path)
        r_img = load_image(r_path)
        lp, s, p = compute_metrics(g_img, r_img)
        lp_list.append(lp)
        s_list.append(s)
        p_list.append(p)
    
    results[gen] = {
        "LPIPS": (np.mean(lp_list), np.std(lp_list)),
        "SSIM": (np.mean(s_list), np.std(s_list)),
        "PSNR": (np.mean(p_list), np.std(p_list))
    }

# ==========================
# 打印和保存 txt
# ==========================
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("CycleGAN 图像质量指标结果\n")
    f.write("====================================\n")
    for k, v in results.items():
        print(f"{k}:")
        f.write(f"{k}:\n")
        for metric, (mean, std) in v.items():
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
            f.write(f"  {metric}: {mean:.4f} ± {std:.4f}\n")
        f.write("------------------------------------\n")
print(f"✅ 指标已保存到: {txt_path}")

# ==========================
# 绘制柱状图并保存
# ==========================
metrics_names = ['LPIPS', 'SSIM', 'PSNR']
fig, axes = plt.subplots(1, len(metrics_names), figsize=(15,5))
for i, metric in enumerate(metrics_names):
    values = [results[c][metric][0] for c, _ in categories]
    stds = [results[c][metric][1] for c, _ in categories]
    axes[i].bar([c for c,_ in categories], values, yerr=stds, capsize=5)
    axes[i].set_title(metric)
    axes[i].set_ylabel(metric)
plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.show()
print(f"✅ 柱状图已保存到: {png_path}")
