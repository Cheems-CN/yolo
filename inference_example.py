"""
推理脚本示例 - 使用训练好的模型对图片进行预测
Inference script example - Using trained model for prediction on images
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yolov11 import YOLOv11


def load_model(model_path, num_classes=7, device='cpu'):
    """
    加载训练好的模型
    Load trained model
    
    Args:
        model_path: 模型权重文件路径 / Path to model weights file
        num_classes: 类别数量 / Number of classes
        device: 设备 / Device
    """
    model = YOLOv11(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型已加载 / Model loaded from: {model_path}")
    return model


def preprocess_image(image_path, img_size=640):
    """
    预处理输入图片
    Preprocess input image
    
    Args:
        image_path: 图片路径 / Image path
        img_size: 目标图片大小 / Target image size
    
    Returns:
        image_tensor: 预处理后的图片tensor / Preprocessed image tensor
        orig_img: 原始图片 / Original image
        scale: 缩放比例 / Scale ratio
    """
    # 加载图片 / Load image
    orig_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = orig_img.size
    
    # Resize图片 / Resize image
    img = orig_img.resize((img_size, img_size))
    
    # 转换为tensor / Convert to tensor
    img_array = np.array(img).astype(np.float32) / 255.0  # 归一化到[0,1] / Normalize to [0,1]
    image_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度 / Add batch dimension
    
    scale = (orig_w / img_size, orig_h / img_size)
    
    return image_tensor, orig_img, scale


def postprocess_predictions(predictions, conf_threshold=0.25, iou_threshold=0.45):
    """
    后处理预测结果
    Post-process predictions
    
    Args:
        predictions: 模型输出 / Model output [1, 4+num_classes, num_anchors]
        conf_threshold: 置信度阈值 / Confidence threshold
        iou_threshold: NMS的IoU阈值 / IoU threshold for NMS
    
    Returns:
        detections: 检测结果列表 / List of detections [x, y, w, h, conf, class_id]
    """
    # predictions shape: [1, 4+num_classes, 8400]
    # 转置为 [1, 8400, 4+num_classes]
    # Transpose to [1, 8400, 4+num_classes]
    predictions = predictions.permute(0, 2, 1)
    
    batch_size = predictions.shape[0]
    detections = []
    
    for i in range(batch_size):
        pred = predictions[i]  # [8400, 4+num_classes]
        
        # 分离boxes和scores / Separate boxes and scores
        boxes = pred[:, :4]  # [8400, 4] - (x, y, w, h)
        scores = pred[:, 4:]  # [8400, num_classes]
        
        # 获取每个anchor的最大类别分数和索引
        # Get max class score and index for each anchor
        class_scores, class_ids = scores.max(dim=1)
        
        # 过滤低置信度预测 / Filter low confidence predictions
        mask = class_scores > conf_threshold
        
        if mask.sum() == 0:
            continue
        
        filtered_boxes = boxes[mask]
        filtered_scores = class_scores[mask]
        filtered_classes = class_ids[mask]
        
        # 简单的NMS (可以使用torchvision.ops.nms来优化)
        # Simple NMS (can use torchvision.ops.nms for optimization)
        keep_indices = []
        indices = torch.argsort(filtered_scores, descending=True)
        
        while len(indices) > 0:
            current = indices[0]
            keep_indices.append(current)
            
            if len(indices) == 1:
                break
            
            current_box = filtered_boxes[current]
            other_boxes = filtered_boxes[indices[1:]]
            
            # 计算IoU / Calculate IoU
            ious = box_iou_simple(current_box.unsqueeze(0), other_boxes)
            
            # 保留IoU小于阈值的boxes / Keep boxes with IoU less than threshold
            indices = indices[1:][ious[0] < iou_threshold]
        
        # 收集检测结果 / Collect detection results
        for idx in keep_indices:
            x, y, w, h = filtered_boxes[idx].tolist()
            conf = filtered_scores[idx].item()
            cls_id = filtered_classes[idx].item()
            detections.append([x, y, w, h, conf, cls_id])
    
    return detections


def box_iou_simple(box1, boxes2):
    """
    简单的IoU计算 (用于NMS)
    Simple IoU calculation (for NMS)
    
    Args:
        box1: [1, 4] (x, y, w, h)
        boxes2: [N, 4] (x, y, w, h)
    
    Returns:
        ious: [1, N]
    """
    # 转换为 xyxy 格式 / Convert to xyxy format
    x1, y1, w1, h1 = box1[0]
    box1_xyxy = torch.tensor([x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2])
    
    boxes2_xyxy = torch.zeros_like(boxes2)
    boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2  # x1
    boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2  # y1
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2  # x2
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2  # y2
    
    # 计算交集 / Calculate intersection
    inter_x1 = torch.max(box1_xyxy[0], boxes2_xyxy[:, 0])
    inter_y1 = torch.max(box1_xyxy[1], boxes2_xyxy[:, 1])
    inter_x2 = torch.min(box1_xyxy[2], boxes2_xyxy[:, 2])
    inter_y2 = torch.min(box1_xyxy[3], boxes2_xyxy[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算并集 / Calculate union
    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    boxes2_area = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    union_area = box1_area + boxes2_area - inter_area
    
    ious = inter_area / (union_area + 1e-6)
    return ious.unsqueeze(0)


def visualize_detections(image, detections, scale, class_names=None, save_path=None):
    """
    可视化检测结果
    Visualize detection results
    
    Args:
        image: 原始图片 / Original image
        detections: 检测结果 / Detection results
        scale: 缩放比例 / Scale ratio
        class_names: 类别名称列表 / List of class names
        save_path: 保存路径 / Save path
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    for det in detections:
        x, y, w, h, conf, cls_id = det
        
        # 转换回原始图片尺寸 / Convert back to original image size
        x_orig = x * scale[0]
        y_orig = y * scale[1]
        w_orig = w * scale[0]
        h_orig = h * scale[1]
        
        # 转换为左上角坐标 / Convert to top-left corner coordinates
        x1 = x_orig - w_orig / 2
        y1 = y_orig - h_orig / 2
        
        # 绘制边界框 / Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), w_orig, h_orig,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签 / Add label
        label = f"{int(cls_id)}: {conf:.2f}"
        if class_names and int(cls_id) < len(class_names):
            label = f"{class_names[int(cls_id)]}: {conf:.2f}"
        
        ax.text(
            x1, y1 - 10, label,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
            fontsize=10, color='white'
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"结果已保存到 / Result saved to: {save_path}")
    
    plt.show()


def main():
    """
    主推理函数 / Main inference function
    """
    # ============= 配置参数 / Configuration =============
    
    model_path = "yolov11_final.pth"  # 模型权重路径 / Path to model weights
    image_path = "test_image.png"     # 测试图片路径 / Path to test image
    num_classes = 7                    # 类别数量 / Number of classes
    img_size = 640                     # 图片大小 / Image size
    conf_threshold = 0.25              # 置信度阈值 / Confidence threshold
    iou_threshold = 0.45               # NMS IoU阈值 / NMS IoU threshold
    
    # 类别名称 (可选) / Class names (optional)
    class_names = [f"Class_{i}" for i in range(num_classes)]
    # 或者自定义类别名称 / Or customize class names:
    # class_names = ["defect_type_1", "defect_type_2", ...]
    
    # 设备 / Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # ============= 加载模型 / Load model =============
    
    print("\n加载模型 / Loading model...")
    model = load_model(model_path, num_classes=num_classes, device=device)
    
    # ============= 预处理图片 / Preprocess image =============
    
    print(f"\n加载图片 / Loading image: {image_path}")
    image_tensor, orig_img, scale = preprocess_image(image_path, img_size=img_size)
    image_tensor = image_tensor.to(device)
    
    print(f"原始图片尺寸 / Original image size: {orig_img.size}")
    print(f"输入图片尺寸 / Input image size: {img_size}x{img_size}")
    
    # ============= 推理 / Inference =============
    
    print("\n开始推理 / Starting inference...")
    with torch.no_grad():
        predictions, _ = model(image_tensor)
    
    print(f"预测输出形状 / Prediction output shape: {predictions.shape}")
    
    # ============= 后处理 / Post-process =============
    
    print("\n后处理预测结果 / Post-processing predictions...")
    detections = postprocess_predictions(
        predictions.cpu(),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    print(f"检测到 {len(detections)} 个目标 / Detected {len(detections)} objects")
    
    # 打印检测结果 / Print detection results
    for i, det in enumerate(detections):
        x, y, w, h, conf, cls_id = det
        print(f"  目标 {i+1} / Object {i+1}: "
              f"类别 / Class={int(cls_id)}, "
              f"置信度 / Confidence={conf:.3f}, "
              f"位置 / Position=(x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})")
    
    # ============= 可视化 / Visualization =============
    
    print("\n可视化结果 / Visualizing results...")
    save_path = "detection_result.png"
    visualize_detections(
        orig_img,
        detections,
        scale,
        class_names=class_names,
        save_path=save_path
    )
    
    print("\n推理完成！/ Inference complete!")


if __name__ == "__main__":
    # 提示用户检查路径 / Remind user to check paths
    print("=" * 80)
    print("请确保: / Please ensure:")
    print("1. 模型权重文件存在 / Model weight file exists")
    print("2. 测试图片路径正确 / Test image path is correct")
    print("3. 类别数量与训练时一致 / Number of classes matches training")
    print("=" * 80)
    print()
    
    main()
