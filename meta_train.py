"""
meta_train.py
基于 train.py 的增强训练脚本，用于记录元信息、消融实验参数、质量筛选和可解释性日志。
此版本训练阶段不调用 quality_filter 和 gradcam_logger，避免显存占用过大。
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp

# ======= Tools =======
from tools.save_metadata import record_metadata
from tools.record_ablations import record_ablations
# quality_filter 和 gradcam_logger 不调用，节省显存

if __name__ == "__main__":
    # ------------------- 1️⃣ 训练参数 -------------------
    opt = TrainOptions().parse()
    opt.num_threads = 0  # Windows 下 DataLoader 安全设置为单线程
    opt.device = init_ddp()

    # ------------------- 2️⃣ 保存元信息 -------------------
    metadata = record_metadata(opt)

    # ------------------- 3️⃣ 保存消融实验参数 -------------------
    ablation_info = record_ablations(opt)

    # ------------------- 4️⃣ 创建数据集 -------------------
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"[Meta] 数据集图片数量 = {dataset_size}")

    # ------------------- 5️⃣ 模型初始化 -------------------
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    # ------------------- 6️⃣ 开始训练 -------------------
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # ------------------- 6.1 显示和保存图像 -------------------
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visuals = model.get_current_visuals()  # 直接获取 tensor，不用 quality_filter
                visualizer.display_current_results(visuals, epoch, total_iters, save_result)

            # ------------------- 6.2 打印和记录损失 -------------------
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            # ------------------- 6.3 保存模型 -------------------
            if total_iters % opt.save_latest_freq == 0:
                print(f"[Meta] 保存最新模型 (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # ------------------- 6.4 更新学习率 -------------------
        model.update_learning_rate()

        # ------------------- 6.5 每个 epoch 保存模型 -------------------
        if epoch % opt.save_epoch_freq == 0:
            print(f"[Meta] 保存 epoch {epoch} 模型, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"[Meta] End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} "
              f"Time Taken: {time.time() - epoch_start_time:.0f} sec")

    # ------------------- 7️⃣ 清理 DDP -------------------
    cleanup_ddp()
