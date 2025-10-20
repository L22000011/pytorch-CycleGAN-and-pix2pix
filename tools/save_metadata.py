import os
import json
import time
import torch
import platform
import random
import numpy as np
from datetime import datetime

def record_metadata(opt):
    """
    记录实验的基本信息：数据集、硬件、参数、随机性、时间等
    保存到 results/<exp_name>/metadata.json
    """
    # 确保目录存在
    save_dir = os.path.join("results", opt.name)
    os.makedirs(save_dir, exist_ok=True)

    # 获取硬件信息
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = str(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)) + " GB" if torch.cuda.is_available() else "N/A"
    cpu_info = platform.processor()

    # 随机种子
    if hasattr(opt, 'seed'):
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt.seed)
        seeds = [opt.seed]
    else:
        seeds = [random.randint(0, 10000)]

    # 数据集统计（简单扫描）
    dataset_path = opt.dataroot
    total_images = 0
    patients = set()
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                total_images += 1
                # 假设 patient_id 是文件夹名的一部分
                patients.add(os.path.basename(root))
    num_patients = len(patients)

    # 生成 metadata 字典
    metadata = {
        "experiment_name": opt.name,
        "dataset": {
            "path": dataset_path,
            "num_images": total_images,
            "num_patients": num_patients,
        },
        "hardware": {
            "gpu_name": gpu_name,
            "gpu_memory": gpu_mem,
            "cpu": cpu_info
        },
        "hyperparams": {
            "learning_rate": opt.lr if hasattr(opt, 'lr') else None,
            "optimizer": "Adam",
            "batch_size": opt.batch_size if hasattr(opt, 'batch_size') else None,
            "lambda_struct": getattr(opt, 'lambda_struct', None),
            "lambda_sym": getattr(opt, 'lambda_sym', None),
        },
        "randomness": {
            "seeds": seeds
        },
        "training": {
            "epochs": opt.n_epochs if hasattr(opt, 'n_epochs') else None,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # 保存 JSON
    save_path = os.path.join(save_dir, "metadata.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"[Metadata] 已保存至: {save_path}")
    return metadata
