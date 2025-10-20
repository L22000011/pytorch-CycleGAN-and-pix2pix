import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from util.visualizer import save_images

if __name__ == '__main__':
    # -------------------- 配置测试参数 --------------------
    opt = TestOptions().parse()  # 使用官方TestOptions
    opt.num_threads = 0   # 测试时关闭多线程
    opt.batch_size = 1    # 测试batch为1
    opt.serial_batches = True  # 不打乱顺序
    opt.no_flip = True    # 不翻转
    opt.display_id = -1   # 不显示窗口
    opt.phase = 'test'
    opt.epoch = 'latest'
    opt.model = 'attention_cycle_gan'

    # -------------------- 创建数据集 --------------------
    dataset = create_dataset(opt)
    print(f"测试数据集大小: {len(dataset)}")

    # -------------------- 创建模型 --------------------
    model = create_model(opt)
    model.setup(opt)
    model.eval()  # 评估模式

    # -------------------- 创建 HTML 保存目录 --------------------
    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.epoch}_{opt.phase}')
    webpage = html.HTML(save_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    # -------------------- 遍历数据集进行测试 --------------------
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()  # 生成图像
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    # -------------------- 保存网页 --------------------
    webpage.save()
    print(f"测试完成，生成结果保存在: {save_dir}")
