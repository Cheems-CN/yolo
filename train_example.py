"""
训练脚本示例 - 使用本地PNG图片和TXT标签文件
Example training script - Using local PNG images and TXT label files
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path

from yolov11 import YOLOv11
from criterion.detect_loss import YoloV11DetectionLoss


class YOLODataset(Dataset):
    """
    自定义数据集类，用于加载PNG图片和对应的TXT标签文件
    Custom dataset class for loading PNG images and corresponding TXT label files
    
    数据集目录结构应该是：
    Dataset directory structure should be:
    dataset/
        images/
            img1.png
            img2.png
            ...
        labels/
            img1.txt
            img2.txt
            ...
    
    标签格式 (每行一个目标):
    Label format (one object per line):
    class_id x_center y_center width height
    
    例如 / Example:
    6 0.535577 0.515385 0.234615 0.230769
    6 0.350962 0.415385 0.419231 0.296154
    """
    
    def __init__(self, image_dir, label_dir, img_size=640, transform=None):
        """
        参数 / Args:
            image_dir: PNG图片目录路径 / Path to PNG images directory
            label_dir: TXT标签目录路径 / Path to TXT labels directory
            img_size: 图片resize的目标大小 / Target image size for resizing
            transform: 可选的数据增强 / Optional data augmentation
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transform = transform
        
        # 获取所有PNG图片路径 / Get all PNG image paths
        self.image_files = sorted(list(self.image_dir.glob("*.png")))
        
        print(f"找到 {len(self.image_files)} 张图片 / Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图片 / Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 加载标签 / Load label
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # 解析每一行: class x y w h
                    # Parse each line: class x y w h
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        boxes.append([cls_id, x, y, w, h])
        
        # 图片预处理 / Image preprocessing
        orig_w, orig_h = image.size
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image).astype(np.float32) / 255.0  # 归一化到[0,1] / Normalize to [0,1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # 转换boxes为tensor / Convert boxes to tensor
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, boxes


def collate_fn(batch):
    """
    自定义collate函数，用于处理不同数量的目标对象
    Custom collate function to handle varying numbers of objects
    """
    images = []
    all_boxes = []
    
    for i, (img, boxes) in enumerate(batch):
        images.append(img)
        # 为每个box添加batch索引 / Add batch index to each box
        if len(boxes) > 0:
            batch_idx = torch.full((len(boxes), 1), i, dtype=torch.float32)
            boxes_with_idx = torch.cat([batch_idx, boxes], dim=1)
            all_boxes.append(boxes_with_idx)
    
    images = torch.stack(images, 0)
    
    if len(all_boxes) > 0:
        all_boxes = torch.cat(all_boxes, 0)
        # 格式: [batch_idx, class, x, y, w, h]
        # Format: [batch_idx, class, x, y, w, h]
        batch_dict = {
            'imgs': images,
            'batch_idx': all_boxes[:, 0],
            'cls': all_boxes[:, 1:2],
            'bboxes': all_boxes[:, 2:6]
        }
    else:
        # 没有目标的情况 / No objects case
        batch_dict = {
            'imgs': images,
            'batch_idx': torch.zeros(0),
            'cls': torch.zeros(0, 1),
            'bboxes': torch.zeros(0, 4)
        }
    
    return batch_dict


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    """
    训练一个epoch
    Train for one epoch
    """
    model.train()
    total_loss = 0
    total_box_loss = 0
    total_cls_loss = 0
    total_dfl_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备上 / Move data to device
        batch['imgs'] = batch['imgs'].to(device)
        batch['batch_idx'] = batch['batch_idx'].to(device)
        batch['cls'] = batch['cls'].to(device)
        batch['bboxes'] = batch['bboxes'].to(device)
        
        # 前向传播 / Forward pass
        optimizer.zero_grad()
        feats = model(batch['imgs'])
        
        # 计算损失 / Calculate loss
        loss, loss_items = loss_fn(feats, batch)
        
        # 反向传播 / Backward pass
        loss.backward()
        optimizer.step()
        
        # 记录损失 / Record losses
        total_loss += loss.item()
        total_box_loss += loss_items[0].item()
        total_cls_loss += loss_items[1].item()
        total_dfl_loss += loss_items[2].item()
        
        # 打印进度 / Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(Box: {loss_items[0].item():.4f}, "
                  f"Cls: {loss_items[1].item():.4f}, "
                  f"DFL: {loss_items[2].item():.4f})")
    
    # 计算平均损失 / Calculate average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_box = total_box_loss / num_batches
    avg_cls = total_cls_loss / num_batches
    avg_dfl = total_dfl_loss / num_batches
    
    print(f"\nEpoch [{epoch}] 平均损失 / Average Loss: {avg_loss:.4f} "
          f"(Box: {avg_box:.4f}, Cls: {avg_cls:.4f}, DFL: {avg_dfl:.4f})\n")
    
    return avg_loss


def main():
    """
    主训练函数 / Main training function
    """
    # ============= 配置参数 / Configuration =============
    
    # 数据集路径 / Dataset paths
    image_dir = "dataset/images"  # 修改为你的图片目录 / Change to your image directory
    label_dir = "dataset/labels"  # 修改为你的标签目录 / Change to your label directory
    
    # 训练参数 / Training parameters
    num_classes = 7  # 类别数量 / Number of classes (根据你的数据集修改 / modify based on your dataset)
    img_size = 640   # 图片大小 / Image size
    batch_size = 4   # 批次大小 / Batch size
    num_epochs = 100 # 训练轮数 / Number of epochs
    learning_rate = 0.001  # 学习率 / Learning rate
    
    # 设备 / Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # ============= 创建数据集和数据加载器 / Create dataset and dataloader =============
    
    print("\n加载数据集 / Loading dataset...")
    dataset = YOLODataset(image_dir, label_dir, img_size=img_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # 根据CPU核心数调整 / Adjust based on CPU cores
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"数据集大小 / Dataset size: {len(dataset)}")
    print(f"批次数量 / Number of batches: {len(dataloader)}")
    
    # ============= 创建模型 / Create model =============
    
    print("\n创建模型 / Creating model...")
    model = YOLOv11(num_classes=num_classes)
    model = model.to(device)
    
    # ============= 创建损失函数和优化器 / Create loss function and optimizer =============
    
    loss_fn = YoloV11DetectionLoss(model, tal_topk=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器 / Learning rate scheduler (可选 / optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # ============= 开始训练 / Start training =============
    
    print("\n开始训练 / Starting training...")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch)
        
        # 更新学习率 / Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率 / Current learning rate: {current_lr:.6f}")
        
        # 保存检查点 / Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f"checkpoints/yolov11_epoch_{epoch}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"保存检查点 / Checkpoint saved: {checkpoint_path}")
        
        print("=" * 80)
    
    # ============= 保存最终模型 / Save final model =============
    
    final_model_path = "yolov11_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n训练完成！模型已保存到 / Training complete! Model saved to: {final_model_path}")


if __name__ == "__main__":
    # 提示用户检查数据集路径 / Remind user to check dataset paths
    print("=" * 80)
    print("请确保: / Please ensure:")
    print("1. 数据集目录结构正确 / Dataset directory structure is correct")
    print("2. PNG图片在 dataset/images/ 目录 / PNG images in dataset/images/ directory")
    print("3. TXT标签在 dataset/labels/ 目录 / TXT labels in dataset/labels/ directory")
    print("4. 标签格式: class x y w h (归一化坐标) / Label format: class x y w h (normalized)")
    print("=" * 80)
    print()
    
    main()
