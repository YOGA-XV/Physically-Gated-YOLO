import torch
import torch.nn.functional as F

def compute_physics_consistency_loss(model, mu_values=[16.00, 102.60, 62.66]):
    """
    计算动态物理一致性损失 (Compute Dynamic Physics-Consistency Loss)
    根据 OPIXray 提取的物理先验 (Physical Priors)，对路由器 (Router) 的门控权重 (Gating Weights) 施加物理惩罚。
    
    参数 (Args):
        model: YOLO26 实例化的主网络模型 (Instantiated YOLO26 main model)
        mu_values: 有机物、金属、混合物的 Hue 先验均值 (Prior mean of Hue for organic, metal, mixed)
    """
    l_physics = 0.0
    valid_blocks_count = 0
    
    # 遍历主网络中所有的子模块 (Iterate through all sub-modules in the main network)
    for m in model.modules():
        if type(m).__name__ == 'PhysicallyGatedMoEBlock':
            # 必须在训练模式下且缓存存在才计算 (Must compute only in training mode when cache exists)
            if not hasattr(m, 'last_routing_weights') or not hasattr(m, 'last_hue_map'):
                continue
            
            # 提取缓存张量 (Extract cached tensors)
            weights = m.last_routing_weights  # (B, 3, H, W)
            hue_map = m.last_hue_map          # (B, 1, H, W)
            
            # --- 步骤 1: 动态生成伪色彩物理掩膜 (Dynamic Pseudo-color Physics Mask Generation) ---
            # 转换先验均值为对齐的张量 (Convert prior means to aligned tensor)
            # [0: 有机物 (Organic), 1: 金属 (Metal), 2: 混合物 (Mixed)]
            mu_tensor = torch.tensor(mu_values, device=hue_map.device).view(1, 3, 1, 1)
            
            # 计算色相的环形最短距离 (Calculate shortest circular distance of Hue)
            diff = torch.abs(hue_map - mu_tensor)
            diff = torch.minimum(diff, 180.0 - diff)
            
            # 生成硬伪标签 (Generate hard pseudo-labels)，按最短距离分配类别索引
            pseudo_labels = torch.argmin(diff, dim=1)  # (B, H, W)
            
            # 将伪标签转化为通道分离的目标掩膜 (Convert pseudo-labels to channel-separated target masks)
            target_masks = F.one_hot(pseudo_labels, num_classes=3).permute(0, 3, 1, 2).float()
            
            # --- 步骤 2: 计算梯度惩罚 (Calculate Gradient Penalty) ---
            # 计算网络预测权重与物理事实掩膜的 L1 损失 (Calculate L1 loss between predicted weights and physical ground truth masks)
            l_physics += F.l1_loss(weights, target_masks)
            valid_blocks_count += 1
            
            # 释放显存，销毁计算图节点的引用 (Release VRAM, destroy references to computation graph nodes)
            del m.last_routing_weights
            del m.last_hue_map
            
    # 取平均以稳定梯度 (Average to stabilize gradients)
    return l_physics / (valid_blocks_count + 1e-6)

if __name__ == '__main__':
    # ==========================================
    # 损失计算单元测试 (Loss Computation Unit Test)
    # ==========================================
    print("=== 测试动态物理一致性损失引擎 (Testing Dynamic Physics Consistency Loss Engine) ===")
    
    # 构建模拟的 MoE Block 和 Model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 动态导入，避免独立运行时的路径错误
            import sys, os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.common_layers.moe_blocks import PhysicallyGatedMoEBlock
            
            self.block1 = PhysicallyGatedMoEBlock(64, 128)
            self.block2 = PhysicallyGatedMoEBlock(128, 256)
            
    model = MockModel()
    model.train()
    
    # 模拟输入并前向传播以产生缓存
    dummy_x1 = torch.randn(2, 64, 32, 32)
    dummy_x2 = torch.randn(2, 128, 16, 16)
    dummy_hue = torch.randint(0, 180, (2, 1, 64, 64)).float()
    
    _ = model.block1(dummy_x1, dummy_hue)
    _ = model.block2(dummy_x2, dummy_hue)
    
    # 测试损失计算逻辑
    try:
        loss_val = compute_physics_consistency_loss(model)
        print(f"✅ 物理一致性损失计算成功 (Physics-Consistency Loss computed successfully): {loss_val.item():.4f}")
        assert loss_val.requires_grad == True, "❌ 损失张量丢失了梯度属性！(Loss tensor lost gradient attribute!)"
        print("✅ 损失张量梯度回传链路完整。(Loss tensor gradient backpropagation link is intact.)")
    except Exception as e:
        print(f"❌ 损失计算崩溃 (Loss computation crashed): {e}")