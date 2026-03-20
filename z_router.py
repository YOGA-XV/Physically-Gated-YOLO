import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicallyGatedRouter(nn.Module):
    """
    Physically-Gated Router (Z-Router)
    跨架构通用的物理门控路由器，结合视觉特征与 HSV 物理先验输出专家权重。
    """
    def __init__(self, in_channels, num_experts=3):
        super(PhysicallyGatedRouter, self).__init__()
        self.num_experts = num_experts
        
        # 1. 视觉特征路由投影层 (轻量级 1x1 卷积)
        self.visual_proj = nn.Conv2d(in_channels, num_experts, kernel_size=1)
        
        # 2. 注入物理先验 (OPIXray 真实统计数据)
        # 顺序: [有机物(橙), 金属(蓝), 混合物(绿)]
        mu_values = [16.00, 102.60, 62.66]
        sigma_values = [2.40, 6.91, 15.92]
        
        # 使用 register_buffer 将物理常数注册为模型的 state_dict
        # 这样它们会自动跟随模型加载到 GPU 上，但不会参与梯度反向传播(保持物理定律不可篡改)
        self.register_buffer('mu', torch.tensor(mu_values).view(1, num_experts, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma_values).view(1, num_experts, 1, 1))
        
        # 3. 可学习的物理核缩放因子 (对应公式中的 \alpha 和 \lambda)
        # 初始权重设为 1.0，让网络在训练中自主决定物理先验的控制力度
        self.physics_scale = nn.Parameter(torch.ones(1, num_experts, 1, 1))
        self.physics_lambda = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, hue_map):
        """
        前向传播
        :param x: 视觉特征图，形状为 (B, C, H_f, W_f)
        :param hue_map: 对应的 Hue 通道图，形状为 (B, 1, H_h, W_h)
        :return: 路由权重，形状为 (B, num_experts, H_f, W_f)
        """
        # --- Step 1: 视觉特征的 Logits ---
        v_logits = self.visual_proj(x)  # (B, 3, H_f, W_f)
        
        # --- Step 2: 物理先验的 Logits (高斯核计算) ---
        # 如果传入的 hue_map 尺寸与特征图不匹配，使用双线性插值进行空间对齐
        if hue_map.shape[-2:] != x.shape[-2:]:
            hue_map = F.interpolate(hue_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        # 根据物理公式: \Phi(H) = \alpha * exp( - (H - \mu)^2 / (2 * \sigma^2) )
        # 由于 Hue 具有环形特性 (0 和 179 是相邻的红色系)，需要计算最短的环形距离
        # 这里为了计算效率，使用绝对距离的近似，或者直接使用标准高斯
        diff = torch.abs(hue_map - self.mu)
        # 处理 Hue 的环形边界 (180 等价于 0)
        diff = torch.minimum(diff, 180.0 - diff) 
        
        p_logits = self.physics_scale * torch.exp(-(diff ** 2) / (2 * (self.sigma ** 2)))
        
        # --- Step 3: 深度融合 (Deep Fusion) ---
        # 将视觉感知与物理先验结合
        combined_logits = v_logits + self.physics_lambda * p_logits
        
        # --- Step 4: 输出归一化的门控权重 ---
        # 在通道维度 (专家维度) 做 Softmax，确保各专家权重之和为 1
        routing_weights = F.softmax(combined_logits, dim=1)
        
        return routing_weights


        """#测试
if __name__ == '__main__':
    # ==========================================
    # Z-Router 单元自检脚本 (Unit Test)
    # ==========================================
    print("🚀 开始测试 Z-Router 模块...")
    
    # 1. 模拟输入数据 (Mock Data)
    batch_size = 4
    in_channels = 256  # 假设这是 YOLO Neck 层某一步的特征通道数
    h, w = 32, 32      # 特征图的空间分辨率
    
    # 模拟视觉特征图: (B, C, H, W)
    dummy_x = torch.randn(batch_size, in_channels, h, w)
    
    # 模拟 Hue 通道图: 尺寸可以和特征图不同，测试动态插值对齐
    # Hue 的物理取值范围是 0-179
    dummy_hue = torch.randint(0, 180, (batch_size, 1, h * 2, w * 2)).float()
    
    # 2. 实例化路由器
    # 假设我们有 3 个专家：有机物、金属、混合物
    router = PhysicallyGatedRouter(in_channels=in_channels, num_experts=3)
    
    # 3. 前向传播测试 (Forward Pass)
    try:
        routing_weights = router(dummy_x, dummy_hue)
        print("✅ 前向传播测试通过！")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        exit()
        
    # 4. 关键特性断言与验证
    print("\n--- 维度与物理特性验证 ---")
    
    # 验证输出维度: 应该是 (Batch, Num_Experts, H, W)
    expected_shape = (batch_size, 3, h, w)
    print(f"期望的权重维度: {expected_shape}")
    print(f"实际的权重维度: {routing_weights.shape}")
    assert routing_weights.shape == expected_shape, "维度对齐失败！"
    
    # 验证 Softmax 守恒: 在专家维度（dim=1）上求和，必须绝对等于 1
    # 取第一个 Batch 的第一个像素点做切片验证
    weight_sum = routing_weights[0, :, 0, 0].sum().item()
    print(f"单像素点专家权重求和 (期望为1.0): {weight_sum:.4f}")
    assert abs(weight_sum - 1.0) < 1e-5, "Softmax 守恒失效，门控逻辑异常！"
    
    # 验证物理先验隔离 (极其重要：物理法则不能被梯度篡改)
    print(f"物理常数 mu 是否需要梯度计算 (期望为False): {router.mu.requires_grad}")
    assert not router.mu.requires_grad, "危险！物理先验被挂载了梯度！"
    
    print(f"自适应权重 physics_lambda 是否需要梯度计算 (期望为True): {router.physics_lambda.requires_grad}")
    assert router.physics_lambda.requires_grad, "警告！自适应权重无法被优化器更新！"
    
    print("\n🎉 全部单元测试通过！Z-Router 随时可以并入主网络。")"""