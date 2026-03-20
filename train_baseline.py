import os
import sys

# 1. 设置项目根目录和基础模型路径
# 强制将 baseline 目录插入 sys.path，确保使用的是原生 ultralytics 代码
project_root = r'G:/Physically-Gated-YOLO'
baseline_root = os.path.join(project_root, 'models', 'baseline')
sys.path.insert(0, baseline_root)

from ultralytics import YOLO

def main():
    # 2. 配置文件路径
    # 使用刚创建的 yolo26_baseline.yaml，它不含 PG-MoE 模块
    model_cfg = os.path.join(baseline_root, 'yolo26_baseline.yaml')
    data_cfg = os.path.join(project_root, 'opixray.yaml')

    # 3. 初始化原生 YOLO 模型
    model = YOLO(model_cfg)

    # 4. 开始基准训练
    # 保持与 exp5 相同的核心参数，确保对照公平
    model.train(
        data        = data_cfg,
        epochs      = 300,
        imgsz       = 640,
        batch       = 16,     # 与 exp5 一致
        device      = 0,
        workers     = 4,
        optimizer   = "SGD",
        lr0         = 0.01,   # 与 exp5 一致
        lrf         = 0.01,
        project     = os.path.join(project_root, "runs", "train"),
        name        = "yolo26_baseline_exp",  # 基准实验命名
        save_period = 10,
        cache       = False,
        patience    = 50,     # Early Stopping
        verbose     = True
    )

if __name__ == "__main__":
    main()
