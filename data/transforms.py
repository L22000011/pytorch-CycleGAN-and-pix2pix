import torchvision.transforms as transforms
from PIL import Image

def get_transform(opt, grayscale=False):
    """Return a composed transform for images."""
    transform_list = []

    # resize 或 scale
    if hasattr(opt, 'load_size'):
        transform_list.append(transforms.Resize((opt.load_size, opt.load_size), Image.BICUBIC))

    # 随机裁剪
    if hasattr(opt, 'crop_size'):
        transform_list.append(transforms.RandomCrop(opt.crop_size))

    # 随机水平翻转
    if not getattr(opt, 'no_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip())

    # 转为 tensor
    if grayscale:
        transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(transforms.ToTensor())

    # 归一化到 [-1, 1]
    transform_list.append(transforms.Normalize((0.5,) * (1 if grayscale else 3),
                                               (0.5,) * (1 if grayscale else 3)))

    return transforms.Compose(transform_list)
