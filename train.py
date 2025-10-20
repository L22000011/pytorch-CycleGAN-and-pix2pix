import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp

if __name__ == "__main__":
    # -------------------- 获取训练参数 --------------------
    opt = TrainOptions().parse()
    opt.device = init_ddp()  # DDP 多卡可选初始化

    # -------------------- 创建数据集 --------------------
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    # -------------------- 创建模型 --------------------
    model = create_model(opt)
    model.setup(opt)  # 加载网络和优化器
    visualizer = Visualizer(opt)
    total_iters = 0

    # -------------------- 开始训练 --------------------
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        # 告诉 dataset 当前 epoch（用于 DistributedSampler）
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        # 设置 model 当前 epoch（用于 ROI loss 判断）
        model.current_epoch = epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # 输入数据
            model.set_input(data)
            # 计算损失并更新网络
            model.optimize_parameters()

            # -------------------- 显示和保存 --------------------
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, iter_start_time)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

        # -------------------- 更新学习率 --------------------
        model.update_learning_rate()

        # -------------------- 保存模型 --------------------
        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

    cleanup_ddp()
