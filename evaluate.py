"""
评估脚本 - 计算mAP50和其他检测指标
Evaluation script - Calculate mAP50 and other detection metrics
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from yolov11 import YOLOv11
from train_example import YOLODataset, collate_fn


def box_iou(box1, box2):
    """
    计算两组boxes之间的IoU
    Calculate IoU between two sets of boxes
    
    Args:
        box1: [N, 4] (x1, y1, x2, y2)
        box2: [M, 4] (x1, y1, x2, y2)
    
    Returns:
        iou: [N, M]
    """
    # 计算交集区域
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算各自的面积
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # 计算并集
    union_area = box1_area[:, None] + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def xywh2xyxy(boxes):
    """转换XYWH到XYXY格式 / Convert XYWH to XYXY format"""
    xyxy = boxes.clone() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def compute_ap(recall, precision):
    """
    计算AP (Average Precision)
    Compute AP from precision and recall curve
    
    使用11点插值法 / Using 11-point interpolation method
    """
    # 在开始和结束添加哨兵值
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # 计算precision包络线
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # 计算recall变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # 计算AP = 曲线下面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def evaluate_model(model, dataloader, device, conf_threshold=0.001, iou_threshold=0.5, num_classes=7):
    """
    评估模型性能
    Evaluate model performance
    
    Returns:
        metrics: dict with mAP, mAP50, precision, recall, etc.
    """
    model.eval()
    
    # 存储所有预测和真实标签
    all_predictions = []  # [(image_id, class_id, confidence, box), ...]
    all_ground_truths = []  # [(image_id, class_id, box), ...]
    
    print("开始评估模型 / Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # 移动数据到设备
            images = batch['imgs'].to(device)
            batch_indices = batch['batch_idx']
            gt_classes = batch['cls']
            gt_boxes = batch['bboxes']  # 归一化的XYWH
            
            # 前向传播
            predictions, _ = model(images)  # [B, 4+C, A]
            
            # 转置: [B, A, 4+C]
            predictions = predictions.permute(0, 2, 1)
            
            batch_size = images.shape[0]
            img_size = images.shape[2]
            
            # 处理每张图片
            for img_idx in range(batch_size):
                global_img_id = batch_idx * batch_size + img_idx
                
                # 获取该图片的预测
                pred = predictions[img_idx]  # [A, 4+C]
                boxes = pred[:, :4]  # [A, 4] XYWH像素坐标
                scores = pred[:, 4:]  # [A, C]
                
                # 获取最大类别和置信度
                class_scores, class_ids = scores.max(dim=1)
                
                # 过滤低置信度
                mask = class_scores > conf_threshold
                if mask.sum() > 0:
                    filtered_boxes = boxes[mask]
                    filtered_scores = class_scores[mask]
                    filtered_classes = class_ids[mask]
                    
                    # 归一化boxes (转为0-1范围)
                    normalized_boxes = filtered_boxes.clone()
                    normalized_boxes[:, [0, 2]] /= img_size  # x, w
                    normalized_boxes[:, [1, 3]] /= img_size  # y, h
                    
                    # 存储预测
                    for box, score, cls_id in zip(normalized_boxes, filtered_scores, filtered_classes):
                        all_predictions.append({
                            'image_id': global_img_id,
                            'class_id': cls_id.item(),
                            'confidence': score.item(),
                            'box': box.cpu().numpy()  # XYWH归一化
                        })
                
                # 获取该图片的GT
                gt_mask = batch_indices == img_idx
                if gt_mask.sum() > 0:
                    img_gt_classes = gt_classes[gt_mask]
                    img_gt_boxes = gt_boxes[gt_mask]  # 已经是归一化的XYWH
                    
                    for cls, box in zip(img_gt_classes, img_gt_boxes):
                        all_ground_truths.append({
                            'image_id': global_img_id,
                            'class_id': int(cls.item()),
                            'box': box.cpu().numpy()  # XYWH归一化
                        })
    
    print(f"收集到 {len(all_predictions)} 个预测和 {len(all_ground_truths)} 个GT")
    
    # 计算每个类别的AP
    aps = []
    precisions = []
    recalls = []
    
    for class_id in range(num_classes):
        # 获取该类别的预测和GT
        class_predictions = [p for p in all_predictions if p['class_id'] == class_id]
        class_ground_truths = [gt for gt in all_ground_truths if gt['class_id'] == class_id]
        
        if len(class_ground_truths) == 0:
            print(f"类别 {class_id}: 没有GT样本")
            continue
        
        if len(class_predictions) == 0:
            print(f"类别 {class_id}: 没有预测")
            aps.append(0.0)
            continue
        
        # 按置信度排序预测
        class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 标记GT是否已被匹配
        gt_matched = {}
        for gt in class_ground_truths:
            img_id = gt['image_id']
            if img_id not in gt_matched:
                gt_matched[img_id] = []
            gt_matched[img_id].append({'box': gt['box'], 'matched': False})
        
        # 计算TP和FP
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        
        for pred_idx, pred in enumerate(class_predictions):
            img_id = pred['image_id']
            pred_box = torch.tensor(pred['box']).unsqueeze(0)  # [1, 4] XYWH
            pred_box_xyxy = xywh2xyxy(pred_box)  # [1, 4] XYXY
            
            if img_id not in gt_matched or len(gt_matched[img_id]) == 0:
                fp[pred_idx] = 1
                continue
            
            # 与该图片的所有GT计算IoU
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt_info in enumerate(gt_matched[img_id]):
                if gt_info['matched']:
                    continue
                
                gt_box = torch.tensor(gt_info['box']).unsqueeze(0)  # [1, 4] XYWH
                gt_box_xyxy = xywh2xyxy(gt_box)  # [1, 4] XYXY
                
                iou = box_iou(pred_box_xyxy, gt_box_xyxy)[0, 0].item()
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # 判断TP或FP
            if max_iou >= iou_threshold and max_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched[img_id][max_gt_idx]['matched'] = True
            else:
                fp[pred_idx] = 1
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算recall和precision
        recall = tp_cumsum / len(class_ground_truths)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # 计算AP
        ap = compute_ap(recall, precision)
        aps.append(ap)
        
        if len(recall) > 0:
            precisions.append(precision[-1])
            recalls.append(recall[-1])
        
        print(f"类别 {class_id}: AP@{iou_threshold:.2f} = {ap:.4f}, "
              f"P = {precision[-1]:.4f}, R = {recall[-1]:.4f}")
    
    # 计算mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    mean_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
    mean_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
    
    metrics = {
        'mAP': mAP,
        'mAP50': mAP if iou_threshold == 0.5 else None,
        'precision': mean_precision,
        'recall': mean_recall,
        'aps_per_class': aps,
        'num_predictions': len(all_predictions),
        'num_ground_truths': len(all_ground_truths)
    }
    
    return metrics


def plot_metrics(metrics_history, save_path='metrics_plot.png'):
    """
    绘制训练过程中的指标变化
    Plot metrics evolution during training
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = list(range(1, len(metrics_history) + 1))
    
    # mAP50
    ax = axes[0, 0]
    mAP50_values = [m.get('mAP50', 0) for m in metrics_history]
    ax.plot(epochs, mAP50_values, 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('mAP@0.5 over Epochs')
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes[0, 1]
    precision_values = [m.get('precision', 0) for m in metrics_history]
    ax.plot(epochs, precision_values, 'g-', linewidth=2, marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision over Epochs')
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[1, 0]
    recall_values = [m.get('recall', 0) for m in metrics_history]
    ax.plot(epochs, recall_values, 'r-', linewidth=2, marker='^')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall over Epochs')
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[1, 1]
    f1_values = []
    for m in metrics_history:
        p = m.get('precision', 0)
        r = m.get('recall', 0)
        f1 = 2 * p * r / (p + r + 1e-6)
        f1_values.append(f1)
    ax.plot(epochs, f1_values, 'm-', linewidth=2, marker='d')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score over Epochs')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"指标图表已保存到 / Metrics plot saved to: {save_path}")
    plt.close()


def plot_pr_curve(precision, recall, ap, save_path='pr_curve.png'):
    """
    绘制Precision-Recall曲线
    Plot Precision-Recall curve
    """
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PR曲线已保存到 / PR curve saved to: {save_path}")
    plt.close()


def main():
    """
    主评估函数 / Main evaluation function
    """
    # ============= 配置参数 / Configuration =============
    
    model_path = "yolov11_final.pth"  # 模型权重路径
    image_dir = "dataset/images"      # 图片目录
    label_dir = "dataset/labels"      # 标签目录
    num_classes = 7                    # 类别数量
    img_size = 640                     # 图片大小 (与训练时一致)
    batch_size = 8                     # 批次大小
    conf_threshold = 0.001             # 置信度阈值 (评估时用较低值)
    iou_threshold = 0.5                # IoU阈值 (mAP50)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # ============= 加载模型 / Load model =============
    
    print("\n加载模型 / Loading model...")
    model = YOLOv11(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型已加载 / Model loaded from: {model_path}")
    
    # ============= 加载数据集 / Load dataset =============
    
    print("\n加载数据集 / Loading dataset...")
    dataset = YOLODataset(image_dir, label_dir, img_size=img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"数据集大小 / Dataset size: {len(dataset)}")
    
    # ============= 评估模型 / Evaluate model =============
    
    print("\n" + "=" * 80)
    print("开始评估 / Starting Evaluation")
    print("=" * 80)
    
    metrics = evaluate_model(
        model, 
        dataloader, 
        device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        num_classes=num_classes
    )
    
    # ============= 打印结果 / Print results =============
    
    print("\n" + "=" * 80)
    print("评估结果 / Evaluation Results")
    print("=" * 80)
    print(f"mAP@0.5:   {metrics['mAP']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-6):.4f}")
    print(f"\n检测数量 / Number of detections: {metrics['num_predictions']}")
    print(f"GT数量 / Number of ground truths: {metrics['num_ground_truths']}")
    
    if len(metrics['aps_per_class']) > 0:
        print("\n各类别AP / AP per class:")
        for class_id, ap in enumerate(metrics['aps_per_class']):
            print(f"  Class {class_id}: {ap:.4f}")
    
    # ============= 保存结果 / Save results =============
    
    results_path = "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n结果已保存到 / Results saved to: {results_path}")
    
    print("\n评估完成！/ Evaluation complete!")


if __name__ == "__main__":
    print("=" * 80)
    print("YOLOv11 模型评估 / YOLOv11 Model Evaluation")
    print("=" * 80)
    print()
    
    main()
