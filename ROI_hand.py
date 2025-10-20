import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import matplotlib.patheffects as path_effects

# ==============================================
# 配置
# ==============================================
#base_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_roi\organized"
base_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_edge\organized"
categories = ["fake_A", "rec_A", "fake_B", "rec_B"]
metrics_dir = os.path.join(base_dir, "metrics")
roi_output_dir = os.path.join(metrics_dir, "ROI_visualization")
os.makedirs(roi_output_dir, exist_ok=True)

# 简化后的手部关键点（左右手各一个）
roi_coords = {
    "left_thumb": (50, 80),
    "right_thumb": (80, 230)
}

# 最大每行显示图片数
max_cols = 5

# ==============================================
# 读取指标数据函数（根据实际 TXT 修改）
# ==============================================
def load_metrics(category):
    """
    返回字典：
    { 'filename': { 'ROI名称': (EdgeAcc, GradSim) } }
    """
    metrics = {}
    folder = os.path.join(base_dir, category)
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            # 生成随机指标，可替换为实际读取 TXT
            metrics[filename] = {joint: (np.random.rand(), np.random.rand()) for joint in roi_coords.keys()}
    return metrics

# ==============================================
# 绘制总图
# ==============================================
for category in categories:
    print(f"🔹 处理 {category} ...")
    metrics = load_metrics(category)
    txt_path = os.path.join(roi_output_dir, f"EdgeGrad_{category}.txt")
    img_color_path = os.path.join(roi_output_dir, f"EdgeGrad_{category}_color.png")
    img_gray_path = os.path.join(roi_output_dir, f"EdgeGrad_{category}_gray.png")

    # 计算行列数
    n_images = len(metrics)
    n_cols = min(max_cols, n_images)
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = np.array(axes).reshape(-1)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Edge Accuracy & Gradient Similarity - {category}\n")
        f.write("="*50 + "\n")

        for ax, (filename, roi_data) in zip(axes, metrics.items()):
            img_path_file = os.path.join(base_dir, category, filename)
            img = Image.open(img_path_file).convert("RGB")
            ax.imshow(img)
            ax.axis("on")

            # 标注 ROI
            for joint_name, (edgeacc, gradsim) in roi_data.items():
                x, y = roi_coords.get(joint_name, (0, 0))
                txt = ax.text(
                    x, y,
                    f"{edgeacc:.2f}/{gradsim:.2f}",
                    color='red',
                    fontsize=12,
                    weight='bold'
                )
                txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
                f.write(f"{filename} [{joint_name}]: EdgeAcc={edgeacc:.4f}, GradSim={gradsim:.4f}\n")
            f.write("\n")

        # 多余子图隐藏
        for ax in axes[len(metrics):]:
            ax.axis("off")

    plt.tight_layout()
    # 保存彩色总图
    plt.savefig(img_color_path, dpi=300)
    print(f"✅ {category} 彩色总图保存: {img_color_path}")

    # 保存灰度总图
    fig_gray, axes_gray = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes_gray = np.array(axes_gray).reshape(-1)
    for ax_gray, (filename, roi_data) in zip(axes_gray, metrics.items()):
        img_path_file = os.path.join(base_dir, category, filename)
        img = Image.open(img_path_file).convert("L")  # 灰度
        ax_gray.imshow(img, cmap='gray')
        ax_gray.axis("on")

        for joint_name, (edgeacc, gradsim) in roi_data.items():
            x, y = roi_coords.get(joint_name, (0, 0))
            txt = ax_gray.text(
                x, y,
                f"{edgeacc:.2f}/{gradsim:.2f}",
                color='red',
                fontsize=12,
                weight='bold'
            )
            txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    for ax_gray in axes_gray[len(metrics):]:
        ax_gray.axis("off")

    plt.tight_layout()
    plt.savefig(img_gray_path, dpi=300)
    plt.close('all')
    print(f"✅ {category} 灰度总图保存: {img_gray_path}")
    print(f"✅ TXT 保存: {txt_path}")

print("🎉 全部处理完成！")
