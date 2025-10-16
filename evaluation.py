# evaluation.py

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 1. 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 2

# --- 定义要评估的数据集路径 ---
TEST_IMAGES_DIR = "data/val_images/"
TEST_MASKS_DIR = "data/val_masks/" # 我们需要真实的Mask来做对比

# --- 定义结果输出路径 ---
OUTPUT_DIR = "evaluation_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True) # 自动创建输出文件夹

# --- 可视化颜色配置 ---
COLOR_MAP = {
    0: (0, 0, 0),       # 背景 (黑色)
    1: (0, 255, 0),     # 预测正确的缺陷 (绿色)
    2: (255, 0, 0),     # 漏检 (红色, 真实Mask有，但预测没有)
    3: (0, 0, 255),     # 误检 (蓝色, 预测有，但真实Mask没有)
}

def mask_to_rgb(mask, color_map):
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

def calculate_metrics(true_mask, pred_mask):
    """计算分割指标"""
    # 确保mask是布尔类型
    true_mask = true_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    # 交集和并集
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    
    # 计算IoU
    iou = (intersection + 1e-6) / (union + 1e-6) # 加一个极小值防止除以0
    
    # 计算精确率和召回率 (检出率)
    true_positives = intersection
    false_positives = np.logical_and(np.logical_not(true_mask), pred_mask).sum()
    false_negatives = np.logical_and(true_mask, np.logical_not(pred_mask)).sum()
    
    precision = (true_positives + 1e-6) / (true_positives + false_positives + 1e-6)
    recall = (true_positives + 1e-6) / (true_positives + false_negatives + 1e-6)
    
    return iou, precision, recall

def main():
    # --- 1. 加载模型 ---
    print("--- Loading model ---")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- 2. 准备数据转换 ---
    transform_for_model = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    image_filenames = os.listdir(TEST_IMAGES_DIR)
    
    # --- 3. 初始化指标列表 ---
    total_iou = []
    total_precision = []
    total_recall = []

    print(f"\n--- Starting evaluation on {len(image_filenames)} images ---")
    # --- 4. 遍历所有测试图片 ---
    for image_name in tqdm(image_filenames):
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        mask_path = os.path.join(TEST_MASKS_DIR, f"{base_name}.png")

        # 读取原始图片和真实的Mask
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if original_image is None or true_mask is None:
            continue

        original_h, original_w = original_image.shape
        
        # 预处理并预测
        transformed = transform_for_model(image=original_image)
        image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prediction = model(image_tensor)
        
        pred_mask_small = torch.argmax(prediction.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
        pred_mask_resized = cv2.resize(pred_mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # --- 5. 计算指标 ---
        iou, precision, recall = calculate_metrics(true_mask, pred_mask_resized)
        total_iou.append(iou)
        total_precision.append(precision)
        total_recall.append(recall)

        # --- 6. 生成并保存可视化结果 ---
        # 创建一个更详细的对比Mask: 0=背景, 1=正确预测(TP), 2=漏检(FN), 3=误检(FP)
        comparison_mask = np.zeros_like(true_mask)
        comparison_mask[(true_mask == 1) & (pred_mask_resized == 1)] = 1 # Green (TP)
        comparison_mask[(true_mask == 1) & (pred_mask_resized == 0)] = 2 # Red (FN)
        comparison_mask[(true_mask == 0) & (pred_mask_resized == 1)] = 3 # Blue (FP)

        color_comparison_mask = mask_to_rgb(comparison_mask, COLOR_MAP)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        overlay_image = cv2.addWeighted(original_image_rgb, 0.6, color_comparison_mask, 0.4, 0)
        
        # 将三张图拼接成一张大图
        h, w, _ = original_image_rgb.shape
        combined_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        combined_image[:, :w, :] = original_image_rgb
        combined_image[:, w:w*2, :] = color_comparison_mask
        combined_image[:, w*2:, :] = overlay_image

        # 添加标题
        cv2.putText(combined_image, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, "Comparison (Green=TP, Red=FN, Blue=FP)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_image, "Overlay Result", (w*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 保存图片
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_result.png")
        cv2.imwrite(output_path, combined_image)

    # --- 7. 打印最终的平均指标 ---
    print("\n--- Evaluation Complete ---")
    print(f"Average IoU (Intersection over Union): {np.mean(total_iou):.4f}")
    print(f"Average Precision: {np.mean(total_precision):.4f}")
    print(f"Average Recall (检出率): {np.mean(total_recall):.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()