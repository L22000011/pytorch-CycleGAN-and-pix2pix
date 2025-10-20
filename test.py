import os
import sys
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

if __name__ == '__main__':
    # 解析参数
    opt = TestOptions().parse()
    opt.num_threads = 0   # 测试时只允许单线程
    opt.batch_size = 1    # 测试时 batch 必须是 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display

    # 创建数据集
    dataset = create_dataset(opt)
    # 创建模型
    model = create_model(opt)
    model.setup(opt)

    # 创建网页保存结果
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_iter{opt.load_iter}'
    print(f'creating web directory {web_dir}')
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    # 推理
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # 限制测试数量
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print(f'processing ({i:04d})-th image... {img_path}')
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
