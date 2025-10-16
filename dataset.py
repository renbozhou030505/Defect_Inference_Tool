# dataset.py (版本 3 - 专门解决 jpg vs png 问题)

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class DefectDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # self.ids 现在只包含不带后缀的文件名
        # 例如，从 "image1.jpg" 中提取出 "image1"
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir)]

        # --- 核心修改点在这里 ---
        # self.images_fps 列表存储所有原始图片的完整路径
        self.images_fps = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
        
        # self.masks_fps 列表根据基础文件名(self.ids)智能地构建Mask的完整路径
        # 它会为每个基础文件名强制添加 ".png" 后缀
        self.masks_fps = [os.path.join(masks_dir, f"{id_}.png") for id_ in self.ids]


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # 通过索引获取对应的图片和Mask路径
        image_path = self.images_fps[index]
        mask_path = self.masks_fps[index]
        
        # 以灰度模式读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- 健壮性检查 ---
        # 如果文件读取失败，给出清晰的错误提示
        if image is None:
            raise FileNotFoundError(f"错误: 无法读取图片文件，请检查路径: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"错误: 无法读取Mask文件，请检查路径: {mask_path}。"
                                    f"确保每个图片都有一个对应的 '.png' 格式的Mask文件。")

        # 应用数据增强和转换
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 将掩码的数据类型转换为long
        mask = mask.long()
        
        return image, mask