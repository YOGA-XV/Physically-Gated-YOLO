# Physically-Gated YOLO

<div align="center">

[!\[Python\](https://img.shields.io/badge/Python-3.8%2B-blue.svg null)](https://www.python.org/)
[!\[PyTorch\](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg null)](https://pytorch.org/)
[!\[License\](https://img.shields.io/badge/License-AGPL--3.0-green.svg null)](LICENSE)

**基于物理门控机制的 X 光安检图像目标检测模型**

[English](#english-version) | [简体中文](#项目简介)

</div>

***

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目架构](#项目架构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [核心模块详解](#核心模块详解)
- [训练指南](#训练指南)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [致谢](#致谢)
- [许可证](#许可证)

***

## 项目简介

Physically-Gated YOLO 是一个专门针对 **X 光安检图像违禁品检测** 优化的目标检测模型。本项目创新性地将 **物理先验知识**（HSV 色彩空间的物质分布规律）融入到深度学习网络中，通过 **物理门控混合专家机制 (Physically-Gated MoE)** 有效解决了 X 光图像中物体遮挡、重叠导致的漏检问题。

### 应用场景

- 🛂 机场/车站安检 X 光图像分析
- 🔒 违禁品自动识别与预警
- 📦 物流包裹安全检查
- 🏥 医学影像分析（具有物理成像规律的领域）

### 目标检测类别

本项目基于 **OPIXray 数据集** 进行训练，支持以下 5 类违禁品检测：

|  类别 |        英文名称       | 描述   |
| :-: | :---------------: | :--- |
|  🔪 |   Folding\_Knife  | 折叠刀  |
|  🔪 |  Straight\_Knife  | 直刀   |
|  ✂️ |      Scissor      | 剪刀   |
| 🛠️ |   Utility\_Knife  | 美工刀  |
|  🔧 | Multi-tool\_Knife | 多功能刀 |

***

## 核心特性

### 🎯 物理先验融合

首次将 X 光成像的物质色彩规律作为可学习的先验知识注入神经网络：

| 物质类型 | Hue 均值 (μ) | 标准差 (σ) | 颜色特征         |
| :--: | :--------: | :-----: | :----------- |
|  有机物 |    16.00   |   2.40  | 橙黄色系（塑料、液体等） |
|  金属  |   102.60   |   6.91  | 深蓝色系（刀具、枪支等） |
|  混合物 |    62.66   |  15.92  | 深绿色系（重叠遮挡区域） |

### 🧠 三大领域专家

针对不同物质特性设计专用特征提取算子：

```
┌─────────────────────────────────────────────────────────────┐
│                    专家分工机制                              │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Organic Expert │  Metal Expert   │     Mixed Expert        │
│    有机物专家    │    金属专家      │      混合物专家          │
├─────────────────┼─────────────────┼─────────────────────────┤
│ 扩张深度可分离卷积 │  标准 3×3 卷积  │  CSAF 双重注意力机制    │
│  (dilation=2)   │                 │  (Channel + Spatial)    │
├─────────────────┼─────────────────┼─────────────────────────┤
│  大面积连续纹理   │  锐利边缘特征   │    重叠遮挡区域处理      │
│  低密度有机物    │  高频金属信息    │    复杂混合特征解耦      │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### ⚡ 端到端可训练

- 物理参数作为可学习权重，网络自主调整物理先验的影响力
- 梯度隔离设计，保证物理损失不会干扰主干特征学习
- 支持 NMS-free 端到端检测（可选）

***

## 项目架构

### 整体网络结构

```
输入图像 (640×640×3)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                    Backbone (YOLO26)                       │
│                                                            │
│  Conv(P1/2) → C3k2 → Conv(P2/4) → C3k2 → Conv(P3/8)       │
│      │                                           │          │
│      ▼                                           ▼          │
│  Conv(P4/16) → C3k2 → Conv(P5/32) → C3k2 → SPPF → C2PSA   │
│                                                            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                   Neck + PG-MoE                            │
│                                                            │
│  Upsample → Concat → PG-MoE(512) ──────────────────────┐  │
│      │                                                 │  │
│      ▼                                                 │  │
│  Upsample → Concat → PG-MoE(256) [P3/8-Small]          │  │
│      │                    │                            │  │
│      ▼                    ▼                            │  │
│  Conv → Concat → PG-MoE(512) [P4/16-Medium]            │  │
│      │                    │                            │  │
│      ▼                    ▼                            │  │
│  Conv → Concat → PG-MoE(1024) [P5/32-Large] ◄──────────┘  │
│                                                            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│              Detect Head (P3, P4, P5 多尺度检测)            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
    检测结果 (BBox + Class + Confidence)
```

### PG-MoE Block 内部结构

```
输入特征 x (B, C, H, W) + 色相图 hue_map (B, 1, H, W)
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌─────────────────────────────┐
│   Z-Router    │      │        专家并行处理          │
│  物理门控路由  │      │                             │
│               │      │  ┌─────────────────────┐   │
│ V_logits +    │      │  │   Organic Expert    │   │
│ λ × P_logits  │      │  │   (扩张卷积)         │   │
│               │      │  └─────────────────────┘   │
│  Softmax 归一  │      │  ┌─────────────────────┐   │
│               │      │  │   Metal Expert      │   │
└───────┬───────┘      │  │   (标准卷积)         │   │
        │              │  └─────────────────────┘   │
        │              │  ┌─────────────────────┐   │
        │              │  │   Mixed Expert      │   │
        │              │  │   (双重注意力)       │   │
        │              │  └─────────────────────┘   │
        │              └─────────────┬───────────────┘
        │                            │
        │      专家输出 (B, 3, C, H, W)
        │                            │
        ▼                            ▼
   路由权重 (B, 3, 1, H, W)    专家特征 (B, 3, C, H, W)
        │                            │
        └──────── 加权融合 ──────────┘
                     │
                     ▼
            融合特征 (B, C, H, W)
```

***

## 环境配置

### 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (推荐)
- 内存: 16GB+ (训练)
- GPU 显存: 8GB+ (训练 640×640 图像)

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/Physically-Gated-YOLO.git
cd Physically-Gated-YOLO

# 2. 创建虚拟环境 (推荐)
conda create -n pg-yolo python=3.10
conda activate pg-yolo

# 3. 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装依赖
pip install ultralytics
pip install opencv-python matplotlib seaborn

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 数据集准备

下载 [OPIXray 数据集](https://github.com/OPIXray-author/OPIXray) 并放置在 `data/raw/` 目录下：

```
data/
└── raw/
    └── OPIXray/
        ├── train/
        │   └── train_image/
        │       ├── xxxxx.jpg
        │       └── xxxxx.txt
        └── test/
            └── test_image/
                ├── xxxxx.jpg
                └── xxxxx.txt
```

***

## 快速开始

### 训练模型

```bash
# 进入项目目录
cd Physically-Gated-YOLO

# 开始训练
python scripts/train.py
```

训练参数可在 `scripts/train.py` 中修改：

```python
CFG = dict(
    model_cfg   = "path/to/yolo26.yaml",  # 模型配置文件
    data_cfg    = "path/to/opixray.yaml", # 数据集配置文件
    epochs      = 300,                     # 训练轮数
    imgsz       = 640,                     # 输入图像尺寸
    batch       = 16,                      # 批次大小
    device      = 0,                       # GPU 设备 ID
    optimizer   = "MuSGD",                 # 优化器
    lr0         = 0.01,                    # 初始学习率
    lrf         = 0.01,                    # 最终学习率系数
    patience    = 50,                      # 早停耐心值
)
```

### 模型推理

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/train/yolo26_pg_moe/weights/best.pt")

# 单张图片推理
results = model.predict("path/to/image.jpg", save=True)

# 批量推理
results = model.predict("path/to/images/", save=True)

# 获取检测结果
for result in results:
    boxes = result.boxes          # 边界框
    probs = result.probs          # 分类概率
    masks = result.masks          # 分割掩码（如果有）
```

### 可视化路由权重

```bash
# 可视化专家路由权重分布
python scripts/visualize_routing.py
```

***

## 核心模块详解

### 1. PhysicallyGatedRouter (Z-Router)

物理门控路由器，实现视觉特征与物理先验的深度融合。

**核心公式**：

$$\text{Routing Weights} = \text{Softmax}(V\_{logits} + \lambda \times P\_{logits})$$

其中：

- $V\_{logits}$：视觉特征投影（1×1 卷积）
- $P\_{logits}$：物理先验高斯核
  $$P\_{logits} = \alpha \times \exp\left(-\frac{(H - \mu)^2}{2\sigma^2}\right)$$
- $\mu, \sigma$：基于 OPIXray 统计的物理常数
- $\alpha, \lambda$：可学习的缩放因子

**代码位置**：`models/common_layers/z_router.py`

```python
class PhysicallyGatedRouter(nn.Module):
    def __init__(self, in_channels, num_experts=3):
        super().__init__()
        self.visual_proj = nn.Conv2d(in_channels, num_experts, kernel_size=1)
        
        # 物理先验常数 (不可训练)
        self.register_buffer('mu', torch.tensor([16.00, 102.60, 62.66]))
        self.register_buffer('sigma', torch.tensor([2.40, 6.91, 15.92]))
        
        # 可学习的缩放因子
        self.physics_scale = nn.Parameter(torch.ones(1, num_experts, 1, 1))
        self.physics_lambda = nn.Parameter(torch.tensor(1.0))
```

### 2. PhysicallyGatedMoEBlock

物理门控专家混合块，整合路由器与三大专家。

**代码位置**：`models/common_layers/moe_blocks.py`

```python
class PhysicallyGatedMoEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.router = PhysicallyGatedRouter(in_channels, num_experts=3)
        self.experts = nn.ModuleList([
            OrganicExpert(in_channels, out_channels),   # 有机物专家
            MetalExpert(in_channels, out_channels),     # 金属专家
            MixedExpert(in_channels, out_channels)      # 混合物专家
        ])
```

### 3. 物理一致性损失

引导网络学习符合物理规律的特征分配。

**代码位置**：`pg_utils/loss_physics.py`

```python
def compute_physics_consistency_loss(model, mu_values=[16.00, 102.60, 62.66]):
    """
    计算动态物理一致性损失
    - 根据色相值动态生成伪标签
    - 计算路由权重与物理事实的 L1 损失
    """
    # 生成硬伪标签
    pseudo_labels = torch.argmin(diff, dim=1)
    target_masks = F.one_hot(pseudo_labels, num_classes=3)
    
    # 计算损失
    loss = F.l1_loss(weights, target_masks)
    return loss
```

***

## 训练指南

### 训练策略

#### 1. 标准训练

```bash
python scripts/train.py
```

#### 2. 分阶段训练

```bash
# Phase 1: 冻结骨干网络，训练 PG-MoE 模块
python scripts/finetune_v2.py --phase 1

# Phase 2: 解冻全部网络，端到端微调
python scripts/finetune_v3.py --phase 2
```

#### 3. 基线对比训练

```bash
python scripts/train_baseline.py
```

### 关键训练技巧

1. **梯度隔离**：物理损失只优化 Router，不反传到 Backbone
   ```python
   routing_weights = self.router(x.detach(), hue_map.detach())
   ```
2. **学习率策略**：使用 MuSGD 优化器，初始学习率 0.01
3. **早停机制**：mAP 连续 50 轮无提升则停止训练
4. **混合精度训练**：自动启用 FP16 加速（需 GPU 支持）

***

## 实验结果

### OPIXray 数据集性能对比

|             模型            | mAP\@50 | mAP\@50:95 |   参数量  | FPS |
| :-----------------------: | :-----: | :--------: | :----: | :-: |
|      YOLO26-Baseline      |    -    |      -     |  2.57M |  -  |
| **Physically-Gated YOLO** |    -    |      -     | \~3.0M |  -  |

*注：详细实验结果请参考* *`runs/`* *目录下的训练日志*

### 消融实验

|        配置        | 说明           | mAP\@50 |
| :--------------: | :----------- | :-----: |
|    Full Model    | 完整 PG-MoE 模型 |    -    |
|    w/o Physics   | 移除物理先验       |    -    |
| w/o Mixed Expert | 移除混合物专家      |    -    |
|     w/o CSAF     | 移除双重注意力      |    -    |

***

## 项目结构

```
Physically-Gated-YOLO/
│
├── 📁 models/                          # 模型定义
│   ├── 📁 baseline/                    # 基线 YOLO26 模型
│   │   ├── 📁 ultralytics/             # Ultralytics 框架
│   │   └── 📄 yolo26_baseline.yaml     # 基线模型配置
│   │
│   ├── 📁 yolo_pg/                     # PG-YOLO 模型
│   │   └── 📁 ultralytics_src/         # 修改后的 Ultralytics 源码
│   │       └── 📁 ultralytics/
│   │           └── 📁 nn/
│   │               └── 📄 tasks.py     # 核心前向传播逻辑
│   │
│   └── 📁 common_layers/               # 核心创新模块
│       ├── 📄 __init__.py
│       ├── 📄 moe_blocks.py            # PG-MoE 块定义
│       └── 📄 z_router.py              # 物理门控路由器
│
├── 📁 scripts/                         # 训练脚本
│   ├── 📄 train.py                     # 主训练脚本
│   ├── 📄 train_baseline.py            # 基线训练脚本
│   ├── 📄 finetune_v2.py               # 微调脚本 v2
│   ├── 📄 finetune_v3.py               # 微调脚本 v3
│   └── 📄 visualize_routing.py         # 路由可视化
│
├── 📁 pg_utils/                        # 工具函数
│   └── 📄 loss_physics.py              # 物理一致性损失
│
├── 📁 data/                            # 数据目录
│   └── 📁 raw/
│       └── 📁 OPIXray/                 # OPIXray 数据集
│
├── 📁 runs/                            # 训练输出
│   ├── 📁 train/                       # 训练结果
│   └── 📁 finetune/                    # 微调结果
│
├── 📄 opixray.yaml                     # 数据集配置
├── 📄 setup.py                         # 安装脚本
├── 📄 Physically_Gated_YOLO_Architecture.md  # 架构文档
└── 📄 README.md                        # 项目说明
```

***

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 框架基础
- [OPIXray](https://github.com/OPIXray-author/OPIXray) - X 光安检数据集
- [YOLO26](https://docs.ultralytics.com/models/yolo26) - 基础检测架构

***

## 许可证

本项目采用 [AGPL-3.0 License](LICENSE) 开源协议。

***

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [Issue](https://github.com/your-username/Physically-Gated-YOLO/issues)
- 发送邮件至: <your-email@example.com>

***

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star ⭐**

</div>

***

## English Version

# Physically-Gated YOLO

<div align="center">

**A Physics-Guided Object Detection Model for X-ray Security Imaging**

</div>

***

## Overview

Physically-Gated YOLO is an object detection model specifically optimized for **prohibited item detection in X-ray security images**. This project innovatively integrates **physical prior knowledge** (material distribution patterns in HSV color space) into deep learning networks through a **Physically-Gated Mixture of Experts (MoE) mechanism**, effectively addressing the missed detection problem caused by object occlusion and overlap in X-ray images.

## Key Features

- 🔬 **Physics-Informed Deep Learning**: First to inject X-ray imaging color patterns as learnable priors
- 🧠 **Domain-Specific Experts**: Three specialized experts for organic, metal, and mixed materials
- ⚡ **End-to-End Trainable**: Learnable physics parameters with gradient isolation
- 🎯 **Occlusion Robust**: Mixed Expert specifically handles overlapping regions

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/Physically-Gated-YOLO.git
cd Physically-Gated-YOLO

# Install dependencies
pip install -r requirements.txt

# Train the model
python scripts/train.py
```

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{physically-gated-yolo,
  author = {Your Name},
  title = {Physically-Gated YOLO: A Physics-Guided Object Detection Model for X-ray Security Imaging},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-username/Physically-Gated-YOLO}
}
```

***

<div align="center">

Made with ❤️ by the Physically-Gated YOLO Team

</div>
