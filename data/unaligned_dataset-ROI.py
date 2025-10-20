import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np

class UnalignedDataset(BaseDataset):
    """
    Dataset class for unaligned/unpaired datasets, with optional ROI masks.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_mask = get_transform(self.opt, grayscale=True)  # mask 总是单通道

    def __getitem__(self, index):
        # 图像索引
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # 打开图像
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # 转 tensor
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # ROI mask
        # 默认 ROI 为全 1，你可以替换成你的 ROI mask 图像路径读取
        A_mask = Image.fromarray(np.ones((A_img.size[1], A_img.size[0]), np.uint8) * 255)
        B_mask = Image.fromarray(np.ones((B_img.size[1], B_img.size[0]), np.uint8) * 255)

        # 转 tensor
        A_mask = self.transform_mask(A_mask)
        B_mask = self.transform_mask(B_mask)

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
            "A_mask": A_mask,
            "B_mask": B_mask
        }

    def __len__(self):
        return max(self.A_size, self.B_size)
