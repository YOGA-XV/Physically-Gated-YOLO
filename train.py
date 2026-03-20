# scripts/train.py
import os
import sys
import logging
import datetime

# ─── 路径注入 ─────────────────────────────────────────────────────────────────
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root       = os.path.dirname(current_script_dir)
ultralytics_src    = os.path.join(project_root, 'models', 'yolo_pg', 'ultralytics_src')

for p in [project_root, ultralytics_src]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ─── 抑制 Ultralytics 冗余文字日志（不影响进度条）────────────────────────────────
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from ultralytics import YOLO

# ─── 训练配置（在这里集中修改参数）──────────────────────────────────────────────
CFG = dict(
    model_cfg   = os.path.join(ultralytics_src, "ultralytics", "cfg", "models", "26", "yolo26.yaml"),
    data_cfg    = os.path.join(project_root, "opixray.yaml"),
    epochs      = 300,
    imgsz       = 640,
    batch       = 16,
    device      = 0,
    workers     = 4,
    optimizer   = "MuSGD",
    lr0         = 0.01,   # YOLO 官方默认值，不应随 batch 线性缩放
    lrf         = 0.01,
    project     = os.path.join(project_root, "runs", "train"),
    name        = "yolo26_pg_moe_musgd",   # 使用 MuSGD 优化器，gamma=0.05
    save_period = 10,
    cache       = False,
    verbose     = True,
    patience    = 50,     # mAP 连续 50 轮无提升则提前停止（Early Stopping）
)

# ─── 打印函数 ─────────────────────────────────────────────────────────────────
W = 62   # 横线宽度

def banner(title: str):
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")

def section(title: str):
    print(f"\n{'─' * W}")
    print(f"  {title}")
    print(f"{'─' * W}")

def kv(key: str, val, w: int = 20):
    print(f"  {key:<{w}}: {val}")

# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main():
    start_time = datetime.datetime.now()

    # 1. 启动横幅
    banner("🚀  Physically-Gated YOLO  ·  训练启动")
    print(f"  时间  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  项目根: {project_root}")

    # 2. 配置检查
    section("📋  训练配置 (Training Config)")
    for f_key in ("model_cfg", "data_cfg"):
        path = CFG[f_key]
        ok   = "✅" if os.path.exists(path) else "❌ 不存在!"
        kv(f_key, f"{ok}  {os.path.relpath(path, project_root)}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

    kv("epochs",      CFG["epochs"])
    kv("imgsz",       CFG["imgsz"])
    kv("batch",       CFG["batch"])
    kv("optimizer",   CFG["optimizer"])
    kv("lr0 / lrf",   f"{CFG['lr0']} / {CFG['lrf']}")
    kv("device",      f"cuda:{CFG['device']}")
    kv("save_period", f"每 {CFG['save_period']} epoch")
    kv("output dir",  os.path.join(CFG["project"], CFG["name"]))

    # 3. 模型初始化
    section("🔧  模型初始化 (Model Init)")
    print("  正在加载 YOLO26-PG-MoE 模型架构...")
    model = YOLO(CFG["model_cfg"], verbose=False)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  ✅ 模型加载成功  参数量: {n_params/1e6:.2f} M")

    # 4. 训练
    section(f"🏋️  开始训练 — {CFG['epochs']} Epochs")
    print()
    model.train(
        data        = CFG["data_cfg"],
        epochs      = CFG["epochs"],
        imgsz       = CFG["imgsz"],
        batch       = CFG["batch"],
        device      = CFG["device"],
        workers     = CFG["workers"],
        optimizer   = CFG["optimizer"],
        lr0         = CFG["lr0"],
        lrf         = CFG["lrf"],
        project     = CFG["project"],
        name        = CFG["name"],
        save_period = CFG["save_period"],
        cache       = CFG["cache"],
        verbose     = CFG["verbose"],
    )

    # 5. 收尾
    elapsed = datetime.datetime.now() - start_time
    banner(f"✅  训练完成  · 总耗时 {str(elapsed).split('.')[0]}")
    out_dir = os.path.join(CFG["project"], CFG["name"])
    print(f"  权重保存在: {out_dir}")
    print()

if __name__ == "__main__":
    main()