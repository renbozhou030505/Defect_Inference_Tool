# check_masks.py
import cv2
import numpy as np
import os

# --- 请修改这个路径为你存放mask的文件夹 ---
MASKS_DIR = "data/train_masks/" 

def check_pixel_values(masks_dir):
    print(f"--- Checking masks in: {masks_dir} ---")
    
    # 遍历文件夹里的所有mask文件
    for filename in os.listdir(masks_dir):
        if filename.endswith(".png"):
            mask_path = os.path.join(masks_dir, filename)
            
            # 以灰度模式读取mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"!! Error: Could not read {filename}")
                continue
                
            # 找到这张mask里所有不重复的像素值
            unique_values = np.unique(mask)
            
            # 打印结果
            print(f"File: {filename} --> Unique pixel values: {unique_values}")
            
            # 检查是否有异常值
            if any(val > 1 for val in unique_values):
                print(f"  ^^^^^^ WARNING: Found pixel value greater than 1 in this mask!")

    print("\n--- Check complete ---")

if __name__ == "__main__":
    check_pixel_values(MASKS_DIR)
    # 你也可以检查验证集的masks
    # check_pixel_values("data/val_masks/")