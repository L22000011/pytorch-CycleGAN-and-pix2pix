# tools/record_ablations.py
import os
import json

def record_ablations(opt):
    """
    记录消融实验参数，例如 lambda_struct, lambda_sym
    保存到 results/<experiment_name>/ablations.json
    """
    save_dir = os.path.join("results", opt.name)
    os.makedirs(save_dir, exist_ok=True)

    ablation_info = {
        "lambda_struct": getattr(opt, "lambda_struct", None),
        "lambda_sym": getattr(opt, "lambda_sym", None),
    }

    save_path = os.path.join(save_dir, "ablations.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(ablation_info, f, indent=4)
    
    print(f"[Ablation] 已保存消融实验参数到: {save_path}")
    return ablation_info
