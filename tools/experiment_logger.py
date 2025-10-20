"""
ExperimentLogger
记录训练日志（时间、重要事件等）
"""

import os
import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.logs = []
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"{timestamp} | {message}")
        print(f"[Log] {timestamp} | {message}")

    def save(self):
        save_path = os.path.join(self.log_dir, "experiment_log.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=4)
        print(f"[Logger] ✅ 已保存训练日志到: {save_path}")
