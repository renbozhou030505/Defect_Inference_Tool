# train.py

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# 从我们自己写的 dataset.py 中导入 DefectDataset 类
from dataset import DefectDataset

# --- 1. 配置参数 (Hyperparameters) ---
# 这些是你可以调整的“旋钮”，用来优化训练过程
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4      # 如果你的GPU显存小，可以调低这个值，比如2或4
NUM_EPOCHS = 50     # 训练的总轮次，可以根据情况增加
IMAGE_HEIGHT = 256  # 调整为适合你数据的尺寸
IMAGE_WIDTH = 256
PIN_MEMORY = True   # 加速数据从CPU到GPU的传输
NUM_CLASSES = 2     # 重要！你的类别总数 = 背景 + 所有缺陷种类。例如，背景+划痕+凹坑+锈蚀 = 4

# --- 2. 数据路径 ---
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

# --- 3. 训练函数 (定义一个训练轮次的逻辑) ---
def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training") # 使用tqdm创建进度条

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # 前向传播 (Forward pass)
        # 使用自动混合精度，可以加速训练并减少显存占用
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # 反向传播 (Backward pass)
        optimizer.zero_grad() # 清空上一轮的梯度
        scaler.scale(loss).backward() # 计算梯度
        scaler.step(optimizer) # 更新模型权重
        scaler.update() # 更新scaler

        # 更新进度条上的损失值
        loop.set_postfix(loss=loss.item())

# --- 4. 验证函数 (用于评估模型在验证集上的表现) ---
def check_performance(loader, model, loss_fn, device):
    model.eval() # 将模型设置为评估模式
    total_loss = 0
    
    with torch.no_grad(): # 在评估时，不需要计算梯度
        loop = tqdm(loader, desc="Validating")
        for data, targets in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    model.train() # 将模型重新设置回训练模式
    return avg_loss


# --- 5. 主函数 (程序的入口) ---
def main():
    # 定义数据增强和转换
    # 训练集使用丰富的增强手段，以提高模型泛化能力
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.5], std=[0.5]), # 针对单通道灰度图归一化
        ToTensorV2(), # 将numpy数组转换为PyTorch张量
    ])

    # 验证集只做最基础的尺寸和归一化处理
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    # ###################################################################### #
    # ##############           这里就是U-Net模型！         ################## #
    # ###################################################################### #
    # 我们使用 segmentation-models-pytorch (smp) 库来轻松构建一个强大的U-Net。
    # - encoder_name="resnet34": 使用ResNet34作为U-Net的编码器（左半边），这比基础U-Net更强大。
    # - encoder_weights="imagenet": 加载在ImageNet上预训练的权重，实现迁移学习，加速收敛，提升效果。
    # - in_channels=1: 明确告诉模型，我们的输入图片是单通道的（灰度图）。
    # - classes=NUM_CLASSES: 告诉模型，最终需要分割出多少个类别。
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=1,                  
        classes=NUM_CLASSES,            
    ).to(DEVICE)
    # ###################################################################### #
    
    # 定义损失函数和优化器
    # CrossEntropyLoss是多分类分割任务的标准选择
    loss_fn = nn.CrossEntropyLoss()
    # Adam是一种高效的优化算法
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 准备数据集和数据加载器
    train_dataset = DefectDataset(images_dir=TRAIN_IMG_DIR, masks_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    
    val_dataset = DefectDataset(images_dir=VAL_IMG_DIR, masks_dir=VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    
    # GradScaler用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float("inf")

    # --- 开始训练循环 ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # 执行一轮训练
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        
        # 在验证集上检查性能
        current_val_loss = check_performance(val_loader, model, loss_fn, DEVICE)
        
        # 如果当前模型的验证损失是历史最低，就保存它
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("=> ✅ Saved new best model!")

# --- 程序的启动点 ---
if __name__ == "__main__":
    main()