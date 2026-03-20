import os
import sys
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

# ─── 路径注入 ─────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ultralytics_src = os.path.join(project_root, 'models', 'yolo_pg', 'ultralytics_src')
sys.path.insert(0, project_root)
sys.path.insert(0, ultralytics_src)

from ultralytics import YOLO

class RoutingVisualizer:
    def __init__(self, model_path, device='cuda:0'):
        """
        初始化可视化器
        :param model_path: .pt 模型权重路径
        :param device: 推理设备
        """
        self.device = torch.device(device)
        self.yolo_model = YOLO(model_path)
        self.model = self.yolo_model.model.to(self.device).eval()
        self.routing_data = {}  # 用于存储 hook 捕获的权重
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        为所有 PhysicallyGatedMoEBlock 注册 forward hook
        """
        from models.common_layers.moe_blocks import PhysicallyGatedMoEBlock
        
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, PhysicallyGatedMoEBlock):
                # 闭包捕获层级名称
                def hook_fn(mod, input, output, layer_name=name):
                    # 获取该模块内部 router 计算出的权重
                    # 也可以从 module.last_routing_weights 获取（如果我们在 forward 里存了）
                    # 这里的 output 是 fused_features，我们实际上需要拦截 router 的输出
                    # 方案：直接拦截 router 的 forward
                    pass
                
                # 重新设计：直接 hook 每一个 block 内部的 router
                def router_hook(mod, input, output, layer_name=name):
                    # output 形状为 (B, 3, H, W)
                    self.routing_data[layer_name] = output.detach().cpu()
                
                module.router.register_forward_hook(router_hook)
                count += 1
        print(f"✅ 成功为 {count} 个 PhysicallyGatedMoEBlock 注册了权重拦截器")

    def visualize(self, img_path, save_dir='runs/vis_routing', alpha=0.6):
        """
        对单张图片进行推理并生成热力图
        """
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.basename(img_path).split('.')[0]
        
        # 1. 预处理原图
        ori_img = cv2.imread(img_path)
        if ori_img is None:
            print(f"❌ 无法读取图片: {img_path}")
            return
        h_orig, w_orig = ori_img.shape[:2]
        
        # 获取 Hue Map 用于模型输入 (如果有必要的话，YOLO 推理时会自动提取)
        # 这里使用 YOLO 的 preprocess
        results = self.yolo_model.predict(img_path, device=self.device) # 执行推理，触发 hooks
        
        # 2. 遍历捕获到的每一层权重
        for layer_name, weights in self.routing_data.items():
            # weights 形状: (1, 3, H, W)
            # 索引 0: 有机物 Organic, 索引 1: 金属 Metal, 索引 2: 混合物 Mixed
            w_org = weights[0, 0].numpy()
            w_met = weights[0, 1].numpy()
            
            # 归一化到 0-255 用于显示
            def to_heatmap(w, base_img):
                # 1. 插值调整尺寸到原图大小
                w_resized = cv2.resize(w, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                # 2. 归一化
                w_norm = (w_resized * 255).astype(np.uint8)
                # 3. 应用伪彩色 (Jet 蓝到红，红代表权重高)
                heatmap = cv2.applyColorMap(w_norm, cv2.COLORMAP_JET)
                # 4. 叠加到原图
                overlay = cv2.addWeighted(base_img, 1 - alpha, heatmap, alpha, 0)
                return overlay

            # 生成并保存
            vis_org = to_heatmap(w_org, ori_img)
            vis_met = to_heatmap(w_met, ori_img)
            
            # 多专家对比展示 (横向拼接)
            combined = np.hstack([ori_img, vis_org, vis_met])
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Organic (W_org) - {layer_name}", (w_orig + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Metal (W_met) - {layer_name}", (w_orig * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            output_path = os.path.join(save_dir, f"{img_name}_{layer_name.replace('.', '_')}.jpg")
            cv2.imwrite(output_path, combined)
            
        print(f"🎉 可视化完成，结果已保存至: {save_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/yolo26_pg_moe_exp5/weights/best.pt')
    parser.add_argument('--source', type=str, required=True, help='图片路径或目录')
    parser.add_argument('--alpha', type=float, default=0.6, help='热力图透明度')
    args = parser.parse_args()

    visualizer = RoutingVisualizer(args.weights)
    
    if os.path.isfile(args.source):
        visualizer.visualize(args.source, alpha=args.alpha)
    else:
        for f in os.listdir(args.source):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                visualizer.visualize(os.path.join(args.source, f), alpha=args.alpha)

if __name__ == '__main__':
    main()
