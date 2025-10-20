import torch
import itertools
from util.image_pool import ImagePool
from models.cyclegan_model import CycleGANModel
from . import networks

class ROICycleGANModel(CycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add ROI-specific options"""
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument('--lambda_ROI', type=float, default=0.05, help='weight for ROI loss')
            parser.add_argument('--roi_start_epoch', type=int, default=5, help='epoch to start ROI loss')
        return parser

    def backward_G(self):
        """Generator loss with optional ROI loss"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_ROI = getattr(self.opt, 'lambda_ROI', 0.0)

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # ROI loss (only after roi_start_epoch)
        if lambda_ROI > 0 and hasattr(self, 'current_epoch') and self.current_epoch >= getattr(self.opt, 'roi_start_epoch', 0):
            # ROI mask must be provided by dataset
            self.loss_ROI_A = torch.nn.functional.l1_loss(self.fake_B * self.A_mask, self.real_B * self.A_mask) * lambda_ROI
            self.loss_ROI_B = torch.nn.functional.l1_loss(self.fake_A * self.B_mask, self.real_A * self.B_mask) * lambda_ROI
        else:
            self.loss_ROI_A = 0
            self.loss_ROI_B = 0

        # Total generator loss
        self.loss_G = (self.loss_G_A + self.loss_G_B +
                       self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_idt_A + self.loss_idt_B +
                       self.loss_ROI_A + self.loss_ROI_B)

        self.loss_G.backward()

    def optimize_parameters(self):
        """Update G and D"""
        self.forward()
        # G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
