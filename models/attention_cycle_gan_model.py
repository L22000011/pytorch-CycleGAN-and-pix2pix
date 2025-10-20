import torch
from .cycle_gan_model import CycleGANModel
from . import networks

""" 时间:10-18 """
import torch.nn.functional as F
import cv2
import numpy as np


class AttentionCycleGANModel(CycleGANModel):
    """CycleGAN + Self-Attention"""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(model='attention_cycle_gan')
        return parser

    def __init__(self, opt):
        """Initialize the Attention CycleGAN model"""
        super(AttentionCycleGANModel, self).__init__(opt)

    def build_generator(self, input_nc, output_nc, ngf, netG, norm_layer, use_dropout, n_blocks):
        """Override generator with attention version"""
        if netG == 'resnet_9blocks':
            return networks.ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                            use_dropout=use_dropout, n_blocks=n_blocks)
        else:
            raise NotImplementedError(f'AttentionCycleGAN only supports resnet_9blocks, got {netG}')
