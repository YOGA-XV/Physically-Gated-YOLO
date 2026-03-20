import torch
import torch.nn as nn
from .z_router import PhysicallyGatedRouter
import torch.nn.functional as F

class OrganicExpert(nn.Module):
    """
    有机物专家 (Organic Expert)
    专注于提取大面积、低密度的连续纹理特征。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用带有膨胀率的深度可分离卷积 (Depthwise Separable Dilated Convolution) 
        # 扩大感受野 (Receptive Field) 的同时保持轻量化
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels, bias=False)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw_conv(self.dw_conv(x))))

class MetalExpert(nn.Module):
    """
    金属专家 (Metal Expert)
    专注于提取锐利边缘和高频信息 (High-frequency Information)。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 标准 3x3 卷积，不使用膨胀，保留局部细节
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn1(self.conv1(x)))

class MixedExpert(nn.Module):
    """
    混合物专家 (Mixed Expert)
    处理复杂的物质重叠 (Material Overlapping) 区域。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 引入通道-空间注意力融合 (Channel-Spatial Attention Fusion, CSAF) 的轻量化变体
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        # 通道注意力 (Channel Attention)
        x = x * self.channel_att(x)
        # 空间注意力 (Spatial Attention)
        spatial_features = torch.cat([torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)], dim=1)
        x = x * self.spatial_att(spatial_features)
        return self.act(x)

class PhysicallyGatedMoEBlock(nn.Module):
    """
    物理引导的门控专家混合块 (Physically-Gated MoE Block)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 实例化路由器 (Router)
        self.router = PhysicallyGatedRouter(in_channels=in_channels, num_experts=3)
        
        # 实例化专家列表 (Expert List)，顺序必须与 z_router 中的物理先验参数顺序一致：
        # [有机物, 金属, 混合物]
        self.experts = nn.ModuleList([
            OrganicExpert(in_channels, out_channels),
            MetalExpert(in_channels, out_channels),
            MixedExpert(in_channels, out_channels)
        ])

    def forward(self, x, hue_map=None):
        """
        :param x: 视觉特征图 (Visual Feature Map) (B, C, H, W)
        :param hue_map: 色相图 (Hue Map) (B, 1, H_orig, W_orig)。
                        若为 None（如 YOLO stride 初始化阶段），则从 x 自动生成替代色相图。
        """
        # 当 hue_map 未提供时（例如 YOLO 框架初始化 stride 时），
        # 用特征图的通道均值归一化代替，保证 forward 可正常运行
        if hue_map is None:
            hue_map = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            hue_map = (hue_map - hue_map.min()) / (hue_map.max() - hue_map.min() + 1e-8)

        # 1. 计算门控权重 (Gating Weights) -> (B, 3, H, W)
        # ⬇️ 梯度隔离：使用 detach() 确保物理损失只优化 Router，不反传到 Backbone
        routing_weights = self.router(x.detach(), hue_map.detach())

        # ⬇️ 新增代码：缓存当前层的路由权重与降采样后的色相图 (Cache weights and hue map)
        if self.training:  # 仅在训练模式下缓存以节省显存
            self.last_routing_weights = routing_weights
            # 使用最近邻插值对齐特征图尺寸
            self.last_hue_map = F.interpolate(hue_map, size=x.shape[-2:], mode='nearest')
        
        # 2. 专家前向传播 (Expert Forward Pass)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(x))
            
        # 将列表堆叠为张量 (B, 3, C, H, W)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 3. 动态特征加权融合 (Dynamic Feature Weighted Fusion)
        # 调整权重维度以支持广播机制: (B, 3, 1, H, W)
        routing_weights = routing_weights.unsqueeze(2)
        
        # 逐元素相乘并按专家维度求和: (B, C, H, W)
        fused_features = torch.sum(expert_outputs * routing_weights, dim=1)
        
        return fused_features


if __name__ == '__main__':
    import torch
    import sys
    import os
    
    # 动态导入处理 (Dynamic Import Handling)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    
    print("=== 测试 PhysicallyGatedMoEBlock 缓存机制 (Testing Caching Mechanism) ===")
    
    # 1. 模拟超参数与张量 (Mock Hyperparameters and Tensors)
    batch_size = 2
    in_channels = 64
    out_channels = 128
    h, w = 32, 32

    block = PhysicallyGatedMoEBlock(in_channels=in_channels, out_channels=out_channels)
    dummy_x = torch.randn(batch_size, in_channels, h, w)
    dummy_hue = torch.randn(batch_size, 1, h * 2, w * 2) 

    # 2. 训练模式测试 (Training Mode Test)
    print("\n--- 训练模式 (Training Mode) ---")
    block.train() # 切换为训练模式
    out_train = block(dummy_x, dummy_hue)
    
    # 断言验证缓存是否存在
    assert hasattr(block, 'last_routing_weights'), "❌ 训练模式下未缓存 routing_weights！(routing_weights not cached in training mode!)"
    assert hasattr(block, 'last_hue_map'), "❌ 训练模式下未缓存 hue_map！(hue_map not cached in training mode!)"
    
    print(f"✅ 成功缓存 routing_weights，维度 (Shape): {block.last_routing_weights.shape}")
    print(f"✅ 成功缓存 hue_map，维度 (Shape): {block.last_hue_map.shape}")
    
    # 断言验证特征图对齐 (Feature Map Alignment)
    # 期望 hue_map 被 F.interpolate 降采样至与 x 相同的空间尺寸 (h, w)
    expected_hue_shape = (batch_size, 1, h, w)
    assert block.last_hue_map.shape == expected_hue_shape, "❌ hue_map 下采样维度不匹配！(hue_map downsampling shape mismatch!)"
    print("✅ 色相图 (Hue Map) 空间维度与视觉特征 (Visual Features) 完美对齐。")

    # 3. 评估模式测试 (Evaluation Mode Test)
    print("\n--- 评估模式 (Evaluation Mode) ---")
    block.eval() # 切换为评估模式
    
    # 手动清除之前的缓存以模拟全新推理环境 (Clear previous cache to simulate fresh inference environment)
    del block.last_routing_weights
    del block.last_hue_map
    
    out_eval = block(dummy_x, dummy_hue)
    
    # 断言验证评估模式下不应存在缓存
    assert not hasattr(block, 'last_routing_weights'), "❌ 致命错误：评估模式下发生了缓存（将导致显存泄漏）！(Fatal error: caching occurred in evaluation mode!)"
    assert not hasattr(block, 'last_hue_map'), "❌ 致命错误：评估模式下发生了缓存！"
    
    print("✅ 评估模式未进行缓存，内存隔离安全。(No caching in evaluation mode, memory isolation safe.)")
    print("\n🎉 缓存机制单元测试全部通过！(All caching mechanism unit tests passed!)")