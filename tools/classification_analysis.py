"""
ClassificationAnalysis
用于在训练/消融实验过程中对分类模型进行分析，例如Accuracy, Precision, Recall
每个 epoch 更新一次
"""

import os
import json

class ClassificationAnalysis:
    def __init__(self, log_dir, variant="baseline"):
        self.log_dir = log_dir
        self.variant = variant
        self.analysis = {}
        os.makedirs(self.log_dir, exist_ok=True)

    def update(self, epoch, model, dataset):
        """
        epoch: 当前轮
        model: 模型对象
        dataset: 数据集对象
        可扩展：调用 model 对测试集进行预测，计算 Accuracy, F1, Sensitivity, Specificity
        """
        # skeleton 示例
        self.analysis[epoch] = {
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1": None
        }

    def save(self):
        save_path = os.path.join(self.log_dir, f"classification_{self.variant}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis, f, indent=4)
        print(f"[Classification] ✅ 已保存分类分析到: {save_path}")
