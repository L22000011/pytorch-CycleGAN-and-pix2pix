import os 
import shutil
from PIL import Image

# ==== 配置 ====
# CycleGAN 生成结果所在的文件夹
#results_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_roi\test_latest\images"
results_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_edge\test_latest\images"
#results_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\IR_CycleGAN_baseline\test_latest\images"
# 输出根目录
#output_root = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\attention_cyclegan\organized"
output_root = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix_multi\results\cyc_edge\organized"
compare_dir = os.path.join(output_root, "comparisons")

# 分类列表
categories = ["real_A", "fake_B", "rec_A", "real_B", "fake_A", "rec_B"]

# ==== 1. 检查 results_dir 是否存在且有图片 ====
if not os.path.exists(results_dir):
    raise FileNotFoundError(f"CycleGAN 测试结果目录不存在: {results_dir}")
if len(os.listdir(results_dir)) == 0:
    raise FileNotFoundError(f"CycleGAN 测试结果目录为空，请先运行 test.py 生成图片: {results_dir}")

# ==== 2. 创建输出目录 ====
os.makedirs(output_root, exist_ok=True)
os.makedirs(compare_dir, exist_ok=True)
for cat in categories:
    os.makedirs(os.path.join(output_root, cat), exist_ok=True)

# ==== 3. 分类整理生成的图片 ====
for file_name in os.listdir(results_dir):
    for cat in categories:
        if file_name.endswith(f"_{cat}.png"):
            src = os.path.join(results_dir, file_name)
            dst = os.path.join(output_root, cat, file_name)
            shutil.copy(src, dst)

# ==== 4. 生成对比图 ====
print("📊 正在生成对比图...")
base_names = {}
for file_name in os.listdir(results_dir):
    for cat in categories:
        if file_name.endswith(f"_{cat}.png"):
            base = file_name.replace(f"_{cat}.png", "")
            if base not in base_names:
                base_names[base] = {}
            base_names[base][cat] = os.path.join(results_dir, file_name)

for base, imgs in base_names.items():
    ordered_imgs = []
    labels = []
    for cat in categories:
        if cat in imgs:
            ordered_imgs.append(Image.open(imgs[cat]))
            labels.append(cat)

    if ordered_imgs:
        widths, heights = zip(*(i.size for i in ordered_imgs))
        total_width = sum(widths)
        max_height = max(heights) + 20  # 上方留出文字空间
        new_im = Image.new("RGB", (total_width, max_height), color=(255,255,255))

        # 在上方添加文字标签
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(new_im)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        x_offset = 0
        for im, label in zip(ordered_imgs, labels):
            draw.text((x_offset + 5, 0), label, fill=(0,0,0), font=font)
            new_im.paste(im, (x_offset, 20))
            x_offset += im.size[0]

        compare_path = os.path.join(compare_dir, f"{base}_comparison.png")
        new_im.save(compare_path)

print(f"✅ 已完成：图片分类 + 对比图生成，结果保存在: {output_root}")
