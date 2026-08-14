from ultralytics.nn.block_fadc import *

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    SPPFCSPC,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C2MC_FEM,TaC2f,C2fCPCA,SPSPPF,
    MC_FEM,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
)

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 可视化特征图的函数
def visualize_feature_maps(model, input_tensor, num_channels=4):
    # 前向传播
    y = [model.cv1(input_tensor)]
    for m in model.m:
        y.append(m(y[-1]))
    y = model.cv2(torch.cat(y, 1))

    # 可视化 cv1 的特征图
    cv1_features = y[0].squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(cv1_features[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle('CV1 Feature Maps')
    plt.show()

    # 可视化 ABlock 的特征图
    for i, block_output in enumerate(y[1:-1]):
        block_features = block_output.squeeze(0).detach().cpu().numpy()
        plt.figure(figsize=(10, 10))
        for j in range(num_channels):
            plt.subplot(1, num_channels, j + 1)
            plt.imshow(block_features[j], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f'ABlock {i + 1} Feature Maps')
        plt.show()

    # 可视化最终输出的特征图
    final_features = y[-1].squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(final_features[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle('Final Output Feature Maps')
    plt.show()


# 加载图片并进行预处理
image_path = r"G:\publicDataSet\MOT\Constr\scane\paper_image\Con-52-3-0.png" # 替换为你的图片路径
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 根据模型输入要求调整尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
input_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度

# 创建模型
model = A2C2f(3, 1024, n=1, a2=True, area=1)  # 输入通道数为 3（RGB）
# model = SFC2f(3, 512, n=1, a2=True, area=1)  # 输入通道数为 3（RGB）

# 可视化特征图
visualize_feature_maps(model, input_tensor)