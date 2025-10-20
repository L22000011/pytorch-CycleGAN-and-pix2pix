"""
CycleGAN Training Script with Lsym (symmetry) and Lstruct (structure/SSIM) losses.
Compatible with original CycleGAN project.

Usage example:
python train_Lsym_Lstruct.py --dataroot ./datasets/IR --name cyclegan_sym_struct --model cycle_gan --batch_size 1 --epoch_count 1 --n_epochs 200 --n_epochs_decay 200
"""

import time
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim  # pip install pytorch-msssim
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


# ----------------------- Auxiliary Loss Functions -----------------------
def compute_lsym(fake):
    """Symmetry loss: horizontal flip and L1 distance"""
    flipped = torch.flip(fake, dims=[-1])
    return F.l1_loss(fake, flipped)


def compute_lstruct(real, fake):
    """Structure loss: SSIM"""
    return 1 - ssim(fake, real, data_range=1.0, size_average=True)


# ----------------------- Training Script -----------------------
if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.device = init_ddp()

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)

    # Handle continue_train case
    if opt.continue_train:
        latest_epoch = getattr(model, "get_latest_epoch", lambda: None)()
        if latest_epoch is not None and opt.epoch_count <= latest_epoch:
            print(
                f"Warning: epoch_count ({opt.epoch_count}) <= latest checkpoint ({latest_epoch}), adjusting to {latest_epoch + 1}"
            )
            opt.epoch_count = latest_epoch + 1

    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        # Dynamic lambdas (can modify if needed)
        lambda_sym_curr = opt.lambda_sym if hasattr(opt, "lambda_sym") else 0.0
        lambda_struct_curr = opt.lambda_struct if hasattr(opt, "lambda_struct") else 0.0

        print(f"[Epoch {epoch}] λ_sym={lambda_sym_curr:.4f}, λ_struct={lambda_struct_curr:.4f}")

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # ---------------- Forward + Optimize ----------------
            model.set_input(data)
            model.optimize_parameters()  # ✅ 内部自动完成 forward/backward/step

            # ---------------- Evaluate Lsym & Lstruct (no backward) ----------------
            with torch.no_grad():
                fake_B = getattr(model, "fake_B", None)
                real_B = getattr(model, "real_B", None)

                if fake_B is not None and real_B is not None:
                    loss_sym = compute_lsym(fake_B)
                    loss_struct = compute_lstruct(real_B, fake_B)
                else:
                    loss_sym = torch.tensor(0.0)
                    loss_struct = torch.tensor(0.0)

            # ---------------- Visualization ----------------
            visual_names = ["real_A", "fake_B", "real_B", "fake_A"]
            if hasattr(model, "idt_A") and model.idt_A is not None:
                visual_names.append("idt_A")
            if hasattr(model, "idt_B") and model.idt_B is not None:
                visual_names.append("idt_B")

            visuals = {name: getattr(model, name) for name in visual_names}
            visualizer.display_current_results(visuals, epoch, total_iters, total_iters % opt.update_html_freq == 0)

            # ---------------- Loss Logging ----------------
            model.losses = {
                "G_total": getattr(model, "loss_G", torch.tensor(0.0)).item(),
                "L_sym": loss_sym.item(),
                "L_struct": loss_struct.item(),
            }

            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, model.losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, model.losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f"Saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # ---------------- Update Learning Rate ----------------
        model.update_learning_rate()

        # ---------------- Epoch Checkpoint ----------------
        if epoch % opt.save_epoch_freq == 0:
            print(f"Saving model at end of epoch {epoch}, total_iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        # ---------------- Debug Print ----------------
        if hasattr(model, "fake_B"):
            print(
                f"[DBG] epoch {epoch} | fake_B min/max/mean: {model.fake_B.min().item():.4f}/"
                f"{model.fake_B.max().item():.4f}/{model.fake_B.mean().item():.4f}"
            )

        print(
            f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec"
        )

    cleanup_ddp()
