# predict.py (版本 2 - 修复了可视化尺寸不匹配的BUG)

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# --- 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
TEST_IMAGE_PATH = "data/val_images/D0-E20-2nd-3.jpg" # 请确保这里是你想要测试的图片
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 2

# 定义颜色映射
COLOR_MAP = {
    0: (0, 0, 0),       # 0=背景(黑色)
    1: (0, 255, 0),     # 1=缺陷(绿色)
}

def mask_to_rgb(mask, color_map):
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

def main():
    # --- 1. 加载模型 ---
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- 2. 图像预处理 ---
    # 这个transform只用于送入模型
    transform_for_model = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    # 读取原始图片，并保存其原始尺寸
    original_image = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Error: Could not read image at {TEST_IMAGE_PATH}")
        return
    
    # 保存原始尺寸，后面会用到
    original_h, original_w = original_image.shape

    # 应用转换，准备送入模型
    transformed = transform_for_model(image=original_image)
    image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    # --- 3. 模型预测 ---
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # --- 4. 结果后处理 ---
    pred_mask_small = torch.argmax(prediction.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)

    # ############################################################### #
    # ##############           核心修复点在这里！         ############# #
    # ############################################################### #
    # 将预测出的小尺寸Mask，缩放回原始图片的大小
    # cv2.INTER_NEAREST 是必须的，因为它能保证像素值不会被模糊，保持为0或1
    pred_mask_resized = cv2.resize(pred_mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    # ############################################################### #

    # --- 5. 缺陷统计 (在缩放后的Mask上进行) ---
    print("--- Defect Statistics ---")
    defect_counts = {}
    for class_id in range(1, NUM_CLASSES):
        class_mask = (pred_mask_resized == class_id).astype(np.uint8)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        count = num_labels - 1
        if count > 0:
            defect_counts[class_id] = count
            print(f"Defect Class {class_id}: Found {count} instances.")
    if not defect_counts:
        print("No defects found.")

    # --- 6. 结果可视化 ---
    # 将缩放后的Mask转换为彩色图
    color_mask = mask_to_rgb(pred_mask_resized, COLOR_MAP)
    
    # 将原始灰度图转为3通道，以便叠加彩色掩码
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # 现在两张图片的尺寸完全一样了，可以安全地叠加
    overlay_image = cv2.addWeighted(original_image_rgb, 0.6, color_mask, 0.4, 0)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(color_mask)
    plt.title("Predicted Mask (Color Coded)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_image)
    plt.title("Overlay Result")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()