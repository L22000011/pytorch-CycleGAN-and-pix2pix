import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# 日志路径
log_file = "checkpoints/IR_CycleGAN/loss_log.txt"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# 读取日志
with open(log_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 解析 loss
records = []
pattern = r"\[Rank 0\] \(epoch: (\d+), iters: (\d+).*?\) , D_A: ([0-9.]+), G_A: ([0-9.]+), cycle_A: ([0-9.]+), idt_A: ([0-9.]+), D_B: ([0-9.]+), G_B: ([0-9.]+), cycle_B: ([0-9.]+), idt_B: ([0-9.]+)"
for line in lines:
    m = re.search(pattern, line)
    if m:
        records.append([int(m.group(2))]+[float(x) for x in m.groups()[2:]])

if not records:
    print("⚠️ 没有解析到任何 loss 数据，请检查 loss_log.txt 格式")
    exit()

# 转为 DataFrame
columns = ["iters","D_A","G_A","cycle_A","idt_A","D_B","G_B","cycle_B","idt_B"]
df = pd.DataFrame(records, columns=columns)

# 平滑函数
def smooth(series, window=10):
    return series.rolling(window, min_periods=1).mean()

# 绘制判别器 loss
plt.figure(figsize=(10,5))
plt.plot(df["iters"], smooth(df["D_A"], window=20), label="D_A", color='tab:blue')
plt.plot(df["iters"], smooth(df["D_B"], window=20), label="D_B", color='tab:orange')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Discriminator Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "discriminator_loss.png"), dpi=300)
plt.close()

# 绘制生成器 loss
plt.figure(figsize=(10,5))
plt.plot(df["iters"], smooth(df["G_A"], window=20), label="G_A", color='tab:blue')
plt.plot(df["iters"], smooth(df["G_B"], window=20), label="G_B", color='tab:orange')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Generator Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "generator_loss.png"), dpi=300)
plt.close()

# 绘制 cycle loss
plt.figure(figsize=(10,5))
plt.plot(df["iters"], smooth(df["cycle_A"], window=20), label="cycle_A", color='tab:blue')
plt.plot(df["iters"], smooth(df["cycle_B"], window=20), label="cycle_B", color='tab:orange')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Cycle-consistency Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cycle_loss.png"), dpi=300)
plt.close()

# 绘制 identity loss
plt.figure(figsize=(10,5))
plt.plot(df["iters"], smooth(df["idt_A"], window=20), label="idt_A", color='tab:blue')
plt.plot(df["iters"], smooth(df["idt_B"], window=20), label="idt_B", color='tab:orange')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Identity Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "identity_loss.png"), dpi=300)
plt.close()

print(f"✅ 所有 loss 曲线已保存到 {save_dir} 文件夹")
