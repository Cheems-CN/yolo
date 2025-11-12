"""
增强训练脚本 - 集成评估和可视化
Enhanced training script - Integrated evaluation and visualization
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import json

from yolov11 import YOLOv11
from criterion.detect_loss import YoloV11DetectionLoss
from train_example import YOLODataset, collate_fn
from evaluate import evaluate_model, plot_metrics


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    """训练一个epoch / Train for one epoch"""
    model.train()
    total_loss = 0
    total_box_loss = 0
    total_cls_loss = 0
    total_dfl_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备上
        batch['imgs'] = batch['imgs'].to(device)
        batch['batch_idx'] = batch['batch_idx'].to(device)
        batch['cls'] = batch['cls'].to(device)
        batch['bboxes'] = batch['bboxes'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        feats = model(batch['imgs'])
        
        # 计算损失
        loss, loss_items = loss_fn(feats, batch)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        total_box_loss += loss_items[0].item()
        total_cls_loss += loss_items[1].item()
        total_dfl_loss += loss_items[2].item()
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(Box: {loss_items[0].item():.4f}, "
                  f"Cls: {loss_items[1].item():.4f}, "
                  f"DFL: {loss_items[2].item():.4f})")
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_box = total_box_loss / num_batches
    avg_cls = total_cls_loss / num_batches
    avg_dfl = total_dfl_loss / num_batches
    
    print(f"\nEpoch [{epoch}] 平均损失 / Average Loss: {avg_loss:.4f} "
          f"(Box: {avg_box:.4f}, Cls: {avg_cls:.4f}, DFL: {avg_dfl:.4f})\n")
    
    return avg_loss, avg_box, avg_cls, avg_dfl


def main():
    """主训练函数 / Main training function"""
    # ============= 配置参数 / Configuration =============
    
    # 数据集路径
    image_dir = "dataset/images"
    label_dir = "dataset/labels"
    
    # 训练参数
    num_classes = 7
    img_size = 640
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.001
    eval_interval = 10  # 每N个epoch评估一次
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # ============= 创建数据集 / Create dataset =============
    
    print("\n加载数据集 / Loading dataset...")
    dataset = YOLODataset(image_dir, label_dir, img_size=img_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建评估数据加载器 (不打乱)
    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # ============= 训练历史记录 / Training history =============
    
    history = {
        'epoch': [],
        'train_loss': [],
        'box_loss': [],
        'cls_loss': [],
        'dfl_loss': [],
        'mAP50': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # ============= 开始训练 / Start training =============
    
    print("\n开始训练 / Starting training...")
    print("=" * 80)
    
    best_mAP = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        avg_loss, avg_box, avg_cls, avg_dfl = train_one_epoch(
            model, dataloader, loss_fn, optimizer, device, epoch
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率 / Current learning rate: {current_lr:.6f}")
        
        # 记录训练损失
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['box_loss'].append(avg_box)
        history['cls_loss'].append(avg_cls)
        history['dfl_loss'].append(avg_dfl)
        
        # 定期评估
        if epoch % eval_interval == 0 or epoch == num_epochs:
            print(f"\n{'='*80}")
            print(f"开始评估 Epoch {epoch} / Starting evaluation for Epoch {epoch}")
            print(f"{'='*80}")
            
            metrics = evaluate_model(
                model,
                eval_dataloader,
                device,
                conf_threshold=0.001,
                iou_threshold=0.5,
                num_classes=num_classes
            )
            
            # 记录评估指标
            history['mAP50'].append(metrics['mAP'])
            history['precision'].append(metrics['precision'])
            history['recall'].append(metrics['recall'])
            
            f1 = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-6)
            history['f1'].append(f1)
            
            print(f"\n评估结果 / Evaluation Results:")
            print(f"  mAP@0.5:   {metrics['mAP']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            
            # 保存最佳模型
            if metrics['mAP'] > best_mAP:
                best_mAP = metrics['mAP']
                best_model_path = "yolov11_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"\n✓ 新的最佳模型！mAP@0.5 = {best_mAP:.4f}")
                print(f"  模型已保存到 / Model saved to: {best_model_path}")
        else:
            # 非评估epoch，用None填充
            history['mAP50'].append(None)
            history['precision'].append(None)
            history['recall'].append(None)
            history['f1'].append(None)
        
        # 保存检查点
        if epoch % 10 == 0:
            checkpoint_path = f"checkpoints/yolov11_epoch_{epoch}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'mAP': history['mAP50'][-1] if history['mAP50'][-1] is not None else 0,
                'history': history
            }, checkpoint_path)
            print(f"保存检查点 / Checkpoint saved: {checkpoint_path}")
        
        print("=" * 80)
    
    # ============= 保存最终模型 / Save final model =============
    
    final_model_path = "yolov11_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n训练完成！模型已保存到 / Training complete! Model saved to: {final_model_path}")
    
    # ============= 保存训练历史 / Save training history =============
    
    history_path = "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到 / Training history saved to: {history_path}")
    
    # ============= 绘制指标 / Plot metrics =============
    
    print("\n生成可视化图表 / Generating visualization plots...")
    
    # 过滤出有评估数据的epoch
    eval_epochs = [e for e, m in zip(history['epoch'], history['mAP50']) if m is not None]
    eval_metrics = []
    for i, epoch in enumerate(history['epoch']):
        if history['mAP50'][i] is not None:
            eval_metrics.append({
                'mAP50': history['mAP50'][i],
                'precision': history['precision'][i],
                'recall': history['recall'][i]
            })
    
    if len(eval_metrics) > 0:
        plot_metrics(eval_metrics, save_path='training_metrics.png')
        print("✓ 指标图表已生成")
    
    print("\n所有任务完成！/ All tasks completed!")


if __name__ == "__main__":
    print("=" * 80)
    print("YOLOv11 训练 (含评估) / YOLOv11 Training (with Evaluation)")
    print("=" * 80)
    print()
    print("请确保: / Please ensure:")
    print("1. 数据集目录结构正确 / Dataset directory structure is correct")
    print("2. PNG图片在 dataset/images/ 目录")
    print("3. TXT标签在 dataset/labels/ 目录")
    print("4. 标签格式: class x y w h (归一化坐标)")
    print("=" * 80)
    print()
    
    main()
