# author: Feng
# contact: 1245272985@qq.com
# datetime:2023/5/24 16:45
# software: PyCharm
"""
文件说明：将特征图可视化

"""
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from backbones.EdgeNeXt import edgenext_base

# 设置设设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 选择预训练模型和目标层名称（在这里，我们使用VGG16和"features.29"）
model = edgenext_base(pretrained=False)
target_layer = 'features.1'

# 加载图像并进行预处理
img_path = "../Data/TrainDataset/image/1.png"
img = Image.open(img_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = preprocess(img)

# 将图像转换为批次形式，并将其输入到模型中以获取特征映射
batch_tensor = torch.unsqueeze(img_tensor, 0)
model.eval()
with torch.no_grad():
    # feature_maps = model(batch_tensor.to(device))[0][target_layer].cpu().detach().numpy()
    feature_maps = model(batch_tensor.to(device))[0].cpu().detach().numpy()

# 对每个通道应用反卷积操作以可视化特征映射
channel_images = []
for i in range(feature_maps.shape[0]):
    channel_image = feature_maps[i]
    channel_image -= channel_image.mean()
    channel_image /= (channel_image.std() + 1e-5)
    channel_image *= 0.1

    # 将特征图可视化到0-1之间的范围
    channel_image += 0.5
    channel_image = np.clip(channel_image, 0, 1)

    # 将特征图缩放为原始图像大小
    channel_image = Image.fromarray(np.uint8(255 * channel_image))
    channel_image = channel_image.resize(img.size)
    channel_images.append(channel_image)

# 创建一个混合图像，将每个通道的可视化结果叠加在一起
mixed_img = Image.new('RGB', img.size)
for i, channel_image in enumerate(channel_images):
    mixed_img = Image.blend(mixed_img, channel_image.convert('RGB'), alpha=0.5)

# 显示可视化结果
mixed_img.show()
