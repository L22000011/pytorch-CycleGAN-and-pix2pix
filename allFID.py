import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
import torch.nn as nn

# ==============================================
# 配置
# ==============================================
# base_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_roi\organized"
base_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_edge\organized"
pairs = [
    ("fake_A", "real_A"),
    ("rec_A", "real_A"),
    ("fake_B", "real_B"),
    ("rec_B", "real_B"),
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用设备: {device}")

# ==============================================
# 图像预处理
# ==============================================
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ==============================================
# 加载图像函数
# ==============================================
def load_images_from_folder(folder, transform):
    imgs = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"⚠️ 跳过 {filename}: {e}")
    return torch.stack(imgs)

# ==============================================
# 获取 InceptionV3 的 pool3 特征（2048维）
# ==============================================
class InceptionV3Features(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        # 保留特征提取部分，不包括分类头
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        # 返回 flatten 后的特征
        features = self.features(x)
        return features.view(features.size(0), -1)


# ==============================================
# 提取特征
# ==============================================
def get_features(model, dataloader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for (x,) in tqdm(dataloader, desc="提取特征中", leave=False):
            x = x.to(device)
            feat = model(x)
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0)

# ==============================================
# 计算FID
# ==============================================
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    model = InceptionV3Features().to(device)

    results = []

    for gen_name, real_name in pairs:
        gen_dir = os.path.join(base_dir, gen_name)
        real_dir = os.path.join(base_dir, real_name)

        if not os.path.exists(gen_dir) or not os.path.exists(real_dir):
            print(f"❌ 找不到路径: {gen_dir} 或 {real_dir}")
            continue

        print(f"\n🔹 正在计算 FID: {gen_name} vs {real_name} ...")

        gen_imgs = load_images_from_folder(gen_dir, transform)
        real_imgs = load_images_from_folder(real_dir, transform)

        print(f"✅ 加载图像数量: {len(gen_imgs)} (生成) vs {len(real_imgs)} (真实)")

        gen_loader = DataLoader(TensorDataset(gen_imgs), batch_size=16, shuffle=False)
        real_loader = DataLoader(TensorDataset(real_imgs), batch_size=16, shuffle=False)

        gen_feats = get_features(model, gen_loader, device)
        real_feats = get_features(model, real_loader, device)

        mu1, sigma1 = np.mean(gen_feats, axis=0), np.cov(gen_feats, rowvar=False)
        mu2, sigma2 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)

        fid = calculate_fid(mu1, sigma1, mu2, sigma2)
        print(f"✅ {gen_name} vs {real_name} → FID = {fid:.4f}")
        results.append((gen_name, real_name, fid))

    print("\n📊 全部结果：")
    for g, r, f in results:
        print(f"   {g:10s} vs {r:10s} → FID = {f:.4f}")
    # ==============================================
    # 保存结果到 txt 文件
    # ==============================================
    save_dir = os.path.join(base_dir, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "FID_baseline.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("📊 FID Baseline Results\n")
        f.write("=========================\n")
        for g, r, fid_value in results:
            f.write(f"{g:10s} vs {r:10s} → FID = {fid_value:.4f}\n")

    print(f"\n✅ 所有结果已保存到：{save_path}")