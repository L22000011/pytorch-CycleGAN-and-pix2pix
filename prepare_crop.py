import cv2
import os

# 原始目录
input_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix\datasets\IR\trainB"
# 新目录
output_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix\datasets\IR\newTrainB"

# 创建新目录
os.makedirs(output_dir, exist_ok=True)

# 遍历所有图片
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图片: {filename}")
            continue

        h, w = img.shape[:2]

        # 找最小边，裁剪中心区域
        min_side = min(h, w)
        start_x = (w - min_side) // 2
        start_y = (h - min_side) // 2
        cropped = img[start_y:start_y+min_side, start_x:start_x+min_side]

        # Resize 到 474x474
        resized = cv2.resize(cropped, (474, 474))

        # 保存到新目录
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, resized)

print("✅ 所有图片已裁剪并保存到 newTrainB！")
import cv2
import os

# 原始 IR 数据集根目录
base_dir = r"E:\Deskbook\pytorch-CycleGAN-and-pix2pix\datasets\IR"

# 要处理的子目录列表
sub_dirs = ["trainA", "trainB", "testA", "testB"]

# 遍历每个子目录
for sub in sub_dirs:
    input_dir = os.path.join(base_dir, sub)
    output_dir = os.path.join(base_dir, f"new{sub}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"正在处理 {input_dir} → {output_dir} ...")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"无法读取图片: {filename}")
                continue

            h, w = img.shape[:2]

            # 中心裁剪
            min_side = min(h, w)
            start_x = (w - min_side) // 2
            start_y = (h - min_side) // 2
            cropped = img[start_y:start_y+min_side, start_x:start_x+min_side]

            # Resize 到 474x474
            resized = cv2.resize(cropped, (474, 474))

            # 保存到新目录
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, resized)

    print(f"✅ {sub} 处理完成，保存到 {output_dir}")

print("🎉 所有目录处理完成！")
