# fix_masks.py

import cv2
import numpy as np
import os

# --- 确认这两个文件夹路径是正确的 ---
DIRS_TO_FIX = ["data/train_masks/", "data/val_masks/"] 

def fix_masks_in_place(mask_dirs):
    """
    这个函数会遍历指定的文件夹，
    读取每一张Mask图片，
    然后把所有非0的像素值都强制设为1。
    最后，它会用修复好的图片覆盖掉原文件。
    """
    for masks_dir in mask_dirs:
        # 检查文件夹是否存在，避免报错
        if not os.path.exists(masks_dir):
            print(f"Warning: Directory not found, skipping: {masks_dir}")
            continue

        print(f"--- Fixing masks in: {masks_dir} ---")
        for filename in os.listdir(masks_dir):
            if filename.endswith(".png"):
                mask_path = os.path.join(masks_dir, filename)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is None:
                    print(f"  Could not read {filename}, skipping.")
                    continue
                
                # --- 核心修复逻辑 ---
                # 这一行代码的意思是：找到mask中所有像素值大于0的位置，
                # 然后把这些位置的像素值全部设置为1。
                mask[mask > 0] = 1
                
                # 原地保存，覆盖旧文件
                cv2.imwrite(mask_path, mask)
                print(f"  Fixed and saved: {filename}")

if __name__ == "__main__":
    fix_masks_in_place(DIRS_TO_FIX)
    print("\n--- Fixing complete! Your masks are now clean. ---")