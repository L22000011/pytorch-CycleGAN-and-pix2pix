import matplotlib.pyplot as plt
import numpy as np

# =======================
# 数据（按照你提供的原始数值）
# =======================
# ROI 点（baseline 在 0.0）
roi_vals = np.array([0.0, 0.05, 0.1, 0.2, 0.5])
labels = ['0', '0.05', '0.1', '0.2', '0.5']

# FID (fake only)
fid_fake_A = np.array([126.1386, 54.8532, 48.0131, 48.0131, 54.6460])
fid_fake_B = np.array([107.7810, 78.4849, 54.7807, 54.7807, 79.4114])

# LPIPS / SSIM / PSNR (fake only)
lpips_fake_A = np.array([0.6747, 0.3831, 0.3881, 0.3810, 0.3876])
lpips_fake_B = np.array([0.6624, 0.4285, 0.3982, 0.4524, 0.4418])

ssim_fake_A = np.array([0.2024, 0.3309, 0.3443, 0.3590, 0.3413])
ssim_fake_B = np.array([0.2697, 0.3412, 0.3427, 0.3344, 0.3364])

psnr_fake_A = np.array([7.0622, 10.4399, 10.4121, 10.7018, 10.3837])
psnr_fake_B = np.array([7.1667, 9.5850, 10.1496, 9.3569, 9.4803])

# 科研色彩
color_A = '#1f77b4'   # 深蓝 - fake_A
color_B = '#ff7f0e'   # 橙色 - fake_B
baseline_marker = dict(marker='*', color='k', s=140, zorder=6)  # baseline 点样式
line_kwargs = dict(linewidth=2, markersize=6)

# ================
# 图 1: FID 比较（突出 baseline）
# ================
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
# 画 ROI>0 的线（不画 baseline 线段，这样 baseline 只用点表示）
ax.plot(roi_vals[1:], fid_fake_A[1:], '-o', color=color_A, label='fake_A (ROI>0)', **line_kwargs)
ax.plot(roi_vals[1:], fid_fake_B[1:], '-s', color=color_B, label='fake_B (ROI>0)', **line_kwargs)
# baseline 单独画点
ax.scatter(roi_vals[0], fid_fake_A[0], **baseline_marker)
ax.scatter(roi_vals[0], fid_fake_B[0], **baseline_marker)
ax.text(0.01, fid_fake_A[0]*1.05, 'Baseline', fontsize=10, color='k')
# 标注最优点（以最小 FID 为优）
best_idx_A = np.argmin(fid_fake_A[1:]) + 1
best_idx_B = np.argmin(fid_fake_B[1:]) + 1
ax.scatter(roi_vals[best_idx_A], fid_fake_A[best_idx_A], facecolors='none', edgecolors='green', s=120, linewidths=2, zorder=7)
ax.scatter(roi_vals[best_idx_B], fid_fake_B[best_idx_B], facecolors='none', edgecolors='green', s=120, linewidths=2, zorder=7)

# 计算并标注百分比改进（相对于 baseline）
improve_A = (fid_fake_A[0] - fid_fake_A[best_idx_A]) / fid_fake_A[0] * 100.0
improve_B = (fid_fake_B[0] - fid_fake_B[best_idx_B]) / fid_fake_B[0] * 100.0
ax.annotate(f'{improve_A:.1f}% ↓ vs baseline', xy=(roi_vals[best_idx_A], fid_fake_A[best_idx_A]),
            xytext=(roi_vals[best_idx_A]+0.01, fid_fake_A[best_idx_A]+5),
            arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)
ax.annotate(f'{improve_B:.1f}% ↓ vs baseline', xy=(roi_vals[best_idx_B], fid_fake_B[best_idx_B]),
            xytext=(roi_vals[best_idx_B]+0.01, fid_fake_B[best_idx_B]+5),
            arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)

ax.set_xticks(roi_vals)
ax.set_xticklabels(labels)
ax.set_xlabel('ROI weight')
ax.set_ylabel('FID (lower is better)')
ax.set_title('FID Comparison: Fake Images (Baseline highlighted)')
ax.grid(alpha=0.35, linestyle='--')
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig('Fig_FID_vs_ROI_baseline_point.png', dpi=300, bbox_inches='tight')

# ================
# 图 2: LPIPS / SSIM / PSNR（每个指标一行，baseline 用点标出）
# ================
fig2, axes = plt.subplots(3, 1, figsize=(7, 9), dpi=120, sharex=True)
# LPIPS
ax = axes[0]
ax.plot(roi_vals[1:], lpips_fake_A[1:], '-o', color=color_A, label='fake_A', **line_kwargs)
ax.plot(roi_vals[1:], lpips_fake_B[1:], '-s', color=color_B, label='fake_B', **line_kwargs)
ax.scatter(roi_vals[0], lpips_fake_A[0], **baseline_marker)
ax.scatter(roi_vals[0], lpips_fake_B[0], **baseline_marker)
ax.set_ylabel('LPIPS (lower better)')
ax.set_title('LPIPS vs ROI (fake images)')
ax.legend(frameon=False)
ax.grid(alpha=0.3, linestyle='--')

# SSIM
ax = axes[1]
ax.plot(roi_vals[1:], ssim_fake_A[1:], '-o', color=color_A, **line_kwargs)
ax.plot(roi_vals[1:], ssim_fake_B[1:], '-s', color=color_B, **line_kwargs)
ax.scatter(roi_vals[0], ssim_fake_A[0], **baseline_marker)
ax.scatter(roi_vals[0], ssim_fake_B[0], **baseline_marker)
ax.set_ylabel('SSIM (higher better)')
ax.set_title('SSIM vs ROI (fake images)')
ax.grid(alpha=0.3, linestyle='--')

# PSNR
ax = axes[2]
ax.plot(roi_vals[1:], psnr_fake_A[1:], '-o', color=color_A, **line_kwargs)
ax.plot(roi_vals[1:], psnr_fake_B[1:], '-s', color=color_B, **line_kwargs)
ax.scatter(roi_vals[0], psnr_fake_A[0], **baseline_marker)
ax.scatter(roi_vals[0], psnr_fake_B[0], **baseline_marker)
ax.set_ylabel('PSNR (higher better)')
ax.set_title('PSNR vs ROI (fake images)')
ax.set_xlabel('ROI weight')
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
fig2.savefig('Fig_metrics_vs_ROI_fake_only_baseline_points.png', dpi=300, bbox_inches='tight')

plt.show()
