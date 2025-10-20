# run_gradcam.py
import torch
from models.cycle_gan_model import CycleGANModel
from tools.gradcam_logger import log_gradcam
from torchvision import transforms
from PIL import Image

# 1. 初始化模型
model = CycleGANModel()
model.setup()  # 加载训练好的模型

# 2. 加载图片
img_path = './datasets/IR/test_image.jpg'
img = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

# 3. 调用 Grad-CAM
log_gradcam(model.netG_A, input_tensor, layer_name='layer4', save_dir='gradcam_results', filename='gradcam_A.png')
log_gradcam(model.netG_B, input_tensor, layer_name='layer4', save_dir='gradcam_results', filename='gradcam_B.png')
