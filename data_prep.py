import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def extract_physics_priors(image_dir):
    """
    遍历 OPIXray 训练集，利用在线统计算法提取不同物理材质的 Hue 通道先验参数，
    输出用于 Z-Router 的 \mu 和 \sigma。
    """
    # 获取目录下所有 jpg 图片
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if not image_paths:
        print("❌ 未找到图片，请检查 OPIXray 路径是否正确！")
        return

    print(f"🚀 开始分析 {len(image_paths)} 张 OPIXray 图像的物理色彩分布...")

    # 使用字典来存储在线统计的累加器 (count, sum, sum_of_squares)
    # 这样可以完美避开 OOM (内存溢出) 问题
    stats = {
        'organic': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0}, # 橙色/有机物
        'metal':   {'count': 0, 'sum': 0.0, 'sum_sq': 0.0}, # 蓝色/金属
        'mixed':   {'count': 0, 'sum': 0.0, 'sum_sq': 0.0}  # 绿色/无机物或重叠
    }

    for path in tqdm(image_paths, desc="Processing Images"):
        img = cv2.imread(path)
        if img is None: 
            continue
        
        # 将 RGB 转换到 HSV 色彩空间 (OpenCV 中 H 的范围是 0-179)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = hsv_img[:, :, 0].astype(np.float64) # 转为 float 防止平方时溢出
        s_channel = hsv_img[:, :, 1]
        v_channel = hsv_img[:, :, 2]

        # 物理遮罩预处理：过滤掉纯白背景、传送带黑色边缘等低饱和度/低亮度无效区域
        valid_mask = (s_channel > 30) & (v_channel > 30)
        valid_hues = h_channel[valid_mask]

        # --- 划定物理材质的经验色相区间 ---
        # 1. 有机物 (橙色系): 跨越 0 的两端
        mask_org = (valid_hues <= 20) | (valid_hues >= 160)
        hues_org = valid_hues[mask_org]
        
        # 2. 金属 (蓝色系): 安检图中最典型的刀具颜色
        mask_metal = (valid_hues >= 90) & (valid_hues <= 130)
        hues_metal = valid_hues[mask_metal]
        
        # 3. 混合物/无机物 (绿色/暗色系): 经常出现在重叠区域
        mask_mixed = (valid_hues >= 35) & (valid_hues <= 85)
        hues_mixed = valid_hues[mask_mixed]

        # --- 更新统计累加器 ---
        if hues_org.size > 0:
            stats['organic']['count'] += hues_org.size
            stats['organic']['sum'] += np.sum(hues_org)
            stats['organic']['sum_sq'] += np.sum(hues_org ** 2)
            
        if hues_metal.size > 0:
            stats['metal']['count'] += hues_metal.size
            stats['metal']['sum'] += np.sum(hues_metal)
            stats['metal']['sum_sq'] += np.sum(hues_metal ** 2)
            
        if hues_mixed.size > 0:
            stats['mixed']['count'] += hues_mixed.size
            stats['mixed']['sum'] += np.sum(hues_mixed)
            stats['mixed']['sum_sq'] += np.sum(hues_mixed ** 2)

    # 计算最终的数学期望 (均值) 和 标准差
    print("\n" + "="*40)
    print("🎯 OPIXray 物理高斯核先验参数 (Z-Router 参数)")
    print("="*40)
    
    for material, name in zip(['organic', 'metal', 'mixed'], ['有机物 (橙色)', '金  属 (蓝色)', '混合物 (绿色)']):
        count = stats[material]['count']
        if count == 0:
            continue
            
        # mu = E[X]
        mu = stats[material]['sum'] / count
        # var = E[X^2] - (E[X])^2
        var = (stats[material]['sum_sq'] / count) - (mu ** 2)
        # 防止因浮点精度导致的极小负数
        sigma = np.sqrt(max(0, var)) 
        
        print(f"[{name}]")
        print(f"  均值 (\mu_k):  {mu:.2f}")
        print(f"  方差 (\sigma_k): {sigma:.2f}\n")

if __name__ == '__main__':
    # 替换为你实际存放 OPIXray 图像的路径 (建议先用 Train 训练集跑)
    OPIXRAY_IMG_DIR = 'data/raw/OPIXray/train/train_image' 
    extract_physics_priors(OPIXRAY_IMG_DIR)