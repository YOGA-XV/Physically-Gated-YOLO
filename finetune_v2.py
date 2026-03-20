"""
═══════════════════════════════════════════════════════════════════════════════
  Physically-Gated YOLO  ·  修复版微调脚本 v2
  
  正确做法：
  1. 从包含 PG-MoE 的 yolo26.yaml 构建完整架构
  2. 将基准模型的兼容权重灌入（MoE 相关层保留随机初始化）
  3. gamma=0，纯 MoE 架构能力对照实验
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import datetime
import torch

# ─── 路径注入 ────────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ultralytics_src = os.path.join(project_root, "models", "yolo_pg", "ultralytics_src")
sys.path.insert(0, ultralytics_src)
sys.path.insert(0, project_root)

from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════════
#  微调配置
# ═══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # ─── PG-MoE 模型架构（从 YAML 构建，包含 MoE 块）────────────────────────
    model_cfg = os.path.join(ultralytics_src, "ultralytics", "cfg", "models", "26", "yolo26.yaml"),
    
    # ─── 基准模型权重（仅灌兼容部分，MoE 层保留随机初始化）────────────────────
    pretrained_weights = os.path.join(
        project_root, "runs", "train", "yolo26_baseline_exp3", "weights", "best.pt"
    ),

    # ─── 数据集 ─────────────────────────────────────────────────────────────
    data_cfg = os.path.join(project_root, "opixray.yaml"),

    # ─── 训练超参数 ──────────────────────────────────────────────────────────
    epochs      = 300,
    imgsz       = 640,
    batch       = 16,
    device      = 0,
    workers     = 4,
    optimizer   = "SGD",
    lr0         = 0.01,       # 从基准权重开始，使用标准学习率
    lrf         = 0.01,       
    warmup_epochs = 3,
    patience    = 50,

    # ─── 输出 ─────────────────────────────────────────────────────────────────
    project     = os.path.join(project_root, "runs", "finetune"),
    name        = "yolo26_pg_moe_ablation_no_physics",
    save_period = 10,
    cache       = False,
    verbose     = True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  权重灌入（仅加载兼容部分）
# ═══════════════════════════════════════════════════════════════════════════════
def load_compatible_weights(model, weights_path):
    """
    将基准模型的权重灌入 PG-MoE 模型。
    - 对于共享的层（Conv, C3k2, SPPF, C2PSA, Detect 等）：直接复制权重
    - 对于 PG-MoE 独有的层（MoE blocks, Router 等）：保持随机初始化
    """
    print(f"\n{'─' * 68}")
    print(f"  📦  加载基准权重 → PG-MoE 模型 (仅兼容部分)")
    print(f"{'─' * 68}")

    # 加载基准模型检查点
    try:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location='cpu')

    # 提取源 state_dict
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt_model = ckpt['model']
        if hasattr(ckpt_model, 'state_dict'):
            src_sd = ckpt_model.float().state_dict()
        else:
            src_sd = ckpt_model
    elif hasattr(ckpt, 'state_dict'):
        src_sd = ckpt.float().state_dict()
    else:
        src_sd = ckpt

    # 目标模型的 state_dict
    dst_sd = model.model.state_dict()

    matched, skipped_shape, new_keys = 0, 0, 0
    new_sd = {}

    for key, dst_param in dst_sd.items():
        if key in src_sd:
            src_param = src_sd[key]
            if src_param.shape == dst_param.shape:
                new_sd[key] = src_param
                matched += 1
            else:
                new_sd[key] = dst_param  # 保持随机初始化
                skipped_shape += 1
        else:
            new_sd[key] = dst_param  # PG-MoE 独有的键，保持随机初始化
            new_keys += 1

    model.model.load_state_dict(new_sd, strict=True)

    total = len(dst_sd)
    print(f"  目标模型键总数      : {total}")
    print(f"  ✅ 成功匹配灌入      : {matched} ({matched/total*100:.1f}%)")
    print(f"  ⚠️  Shape 不匹配     : {skipped_shape}")
    print(f"  🆕 PG-MoE 新增键     : {new_keys} (保持随机初始化)")
    
    return matched, new_keys


# ═══════════════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    start_time = datetime.datetime.now()

    W = 68
    print(f"\n{'═' * W}")
    print(f"  🔬  PG-MoE 对照实验 · gamma=0 (物理损失关闭)")
    print(f"{'═' * W}")
    print(f"  时间  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  项目根: {project_root}")
    print()

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  第 1 步：从 PG-MoE YAML 构建完整模型架构                            ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    print(f"{'─' * W}")
    print(f"  🔧  模型初始化 (从 yolo26.yaml 构建 PG-MoE 架构)")
    print(f"{'─' * W}")
    
    model = YOLO(CFG["model_cfg"], verbose=True)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  ✅ PG-MoE 模型构建成功  参数量: {n_params / 1e6:.2f} M")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  第 2 步：将基准权重灌入（仅兼容部分）                                ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    matched, new_keys = load_compatible_weights(model, CFG["pretrained_weights"])

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  第 3 步：启动训练                                                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    print(f"\n{'─' * W}")
    print(f"  🏋️  开始训练 — {CFG['epochs']} Epochs")
    print(f"  📌 物理损失 gamma = 0 (完全关闭)")
    print(f"  📌 PG-MoE 架构 + 基准兼容权重 + MoE 从零学习")
    print(f"{'─' * W}")
    print()

    model.train(
        data          = CFG["data_cfg"],
        epochs        = CFG["epochs"],
        imgsz         = CFG["imgsz"],
        batch         = CFG["batch"],
        device        = CFG["device"],
        workers       = CFG["workers"],
        optimizer     = CFG["optimizer"],
        lr0           = CFG["lr0"],
        lrf           = CFG["lrf"],
        warmup_epochs = CFG["warmup_epochs"],
        project       = CFG["project"],
        name          = CFG["name"],
        save_period   = CFG["save_period"],
        cache         = CFG["cache"],
        verbose       = CFG["verbose"],
        patience      = CFG["patience"],
    )

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  第 4 步：收尾                                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    elapsed = datetime.datetime.now() - start_time
    print(f"\n{'═' * W}")
    print(f"  ✅  训练完成  · 总耗时 {str(elapsed).split('.')[0]}")
    print(f"{'═' * W}")
    out_dir = os.path.join(CFG["project"], CFG["name"])
    print(f"  权重保存在: {out_dir}")
    print()


if __name__ == "__main__":
    main()
