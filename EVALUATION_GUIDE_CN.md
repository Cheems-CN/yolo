# 评估指标指南 / Evaluation Metrics Guide

## 概述 / Overview

本指南详细说明YOLOv11模型的评估指标计算和可视化功能。

## 评估指标 / Evaluation Metrics

### 1. mAP@0.5 (Mean Average Precision at IoU 0.5)

**定义**: 在IoU阈值为0.5时，所有类别的平均精度

**计算方法**:
1. 对每个类别计算AP (Average Precision)
2. 取所有类别AP的平均值

**公式**:
```
mAP@0.5 = (Σ AP_i) / N
其中 i 为类别索引，N 为类别总数
```

**意义**:
- 衡量模型整体检测性能的标准指标
- 值越高越好，范围 [0, 1]
- 0.5表示预测框与GT的IoU至少为50%

**典型值**:
- > 0.5: 好
- > 0.7: 很好
- > 0.9: 优秀

### 2. AP (Average Precision) - 单类别

**定义**: 单个类别的Precision-Recall曲线下面积

**计算步骤**:
1. 按置信度降序排列该类别的所有预测
2. 对每个预测，判断是TP还是FP (基于IoU阈值)
3. 计算累积TP和FP
4. 计算每个阈值下的Precision和Recall
5. 绘制PR曲线
6. 计算曲线下面积 (使用11点插值法)

**公式**:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
AP = ∫[0,1] Precision(Recall) dRecall
```

### 3. Precision (精确率)

**定义**: 所有检测中正确检测的比例

**公式**:
```
Precision = TP / (TP + FP)
```

**意义**:
- 衡量模型预测的准确性
- 高Precision意味着很少误检
- 值越高越好，范围 [0, 1]

**示例**:
- Precision = 0.9 表示90%的检测是正确的

### 4. Recall (召回率)

**定义**: 所有GT中被正确检测的比例

**公式**:
```
Recall = TP / (TP + FN)
```

**意义**:
- 衡量模型检测的完整性
- 高Recall意味着很少漏检
- 值越高越好，范围 [0, 1]

**示例**:
- Recall = 0.85 表示85%的目标被检测到

### 5. F1 Score

**定义**: Precision和Recall的调和平均

**公式**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**意义**:
- 综合评估Precision和Recall
- 平衡检测准确性和完整性
- 值越高越好，范围 [0, 1]

### 6. IoU (Intersection over Union)

**定义**: 预测框和GT框的交集与并集之比

**公式**:
```
IoU = Area(Pred ∩ GT) / Area(Pred ∪ GT)
```

**阈值说明**:
- IoU ≥ 0.5: 预测被认为是TP (mAP@0.5)
- IoU < 0.5: 预测被认为是FP

---

## 使用方法 / Usage

### 方法1: 独立评估

**适用场景**: 训练完成后评估模型

```bash
python evaluate.py
```

**配置参数** (在`evaluate.py`的`main()`函数中):
```python
model_path = "yolov11_final.pth"  # 模型路径
image_dir = "dataset/images"      # 图片目录
label_dir = "dataset/labels"      # 标签目录
num_classes = 7                    # 类别数量
img_size = 640                     # 图片大小
conf_threshold = 0.001             # 置信度阈值 (评估时用低值)
iou_threshold = 0.5                # IoU阈值 (mAP50)
```

**输出文件**:
- `evaluation_results.json`: 完整的评估结果
  ```json
  {
    "mAP": 0.6234,
    "mAP50": 0.6234,
    "precision": 0.7821,
    "recall": 0.6543,
    "aps_per_class": [0.61, 0.58, 0.67, ...],
    "num_predictions": 1234,
    "num_ground_truths": 1000
  }
  ```

**控制台输出**:
```
评估结果 / Evaluation Results
================================================================================
mAP@0.5:   0.6234
Precision: 0.7821
Recall:    0.6543
F1 Score:  0.7123

各类别AP / AP per class:
  Class 0: 0.6123
  Class 1: 0.5834
  Class 2: 0.6721
  ...
```

### 方法2: 训练时评估

**适用场景**: 训练期间监控模型性能

```bash
python train_with_eval.py
```

**配置参数**:
```python
eval_interval = 10  # 每10个epoch评估一次
```

**输出文件**:
1. **yolov11_best.pth**: 基于mAP的最佳模型
2. **training_history.json**: 完整训练历史
   ```json
   {
     "epoch": [1, 2, 3, ...],
     "train_loss": [45.3, 42.1, 38.5, ...],
     "mAP50": [null, null, 0.45, null, ...],
     "precision": [null, null, 0.68, null, ...],
     "recall": [null, null, 0.54, null, ...]
   }
   ```
3. **training_metrics.png**: 指标演化图表

**可视化图表** (4个子图):
1. mAP@0.5 vs Epoch
2. Precision vs Epoch
3. Recall vs Epoch
4. F1 Score vs Epoch

---

## 评估流程详解 / Evaluation Process

### Step 1: 数据加载

```python
dataset = YOLODataset(image_dir, label_dir, img_size)
dataloader = DataLoader(dataset, batch_size, shuffle=False)
```

### Step 2: 模型推理

```python
model.eval()
with torch.no_grad():
    predictions, _ = model(images)
```

### Step 3: 收集预测和GT

对每张图片:
- 收集所有预测: `{image_id, class_id, confidence, box}`
- 收集所有GT: `{image_id, class_id, box}`

### Step 4: 计算每个类别的AP

对每个类别:
1. 筛选该类别的预测和GT
2. 按置信度排序预测
3. 对每个预测计算IoU
4. 标记TP/FP (IoU ≥ 0.5为TP)
5. 计算Precision和Recall曲线
6. 计算AP (曲线下面积)

### Step 5: 汇总结果

```python
mAP = mean(aps_per_class)
mean_precision = mean(precisions)
mean_recall = mean(recalls)
f1 = 2 * P * R / (P + R)
```

---

## 指标解读 / Metrics Interpretation

### 场景1: 高Precision，低Recall

```
Precision: 0.95
Recall:    0.60
```

**含义**: 
- 模型很少误检，但漏检较多
- 预测的框大多是对的，但很多目标没检测到

**可能原因**:
- 置信度阈值设置过高
- 模型过于保守

**改进方向**:
- 降低置信度阈值
- 增加训练数据
- 调整anchor尺寸

### 场景2: 低Precision，高Recall

```
Precision: 0.55
Recall:    0.92
```

**含义**:
- 模型检测到了大部分目标，但误检很多
- 很多假阳性

**可能原因**:
- 置信度阈值设置过低
- 模型过于激进

**改进方向**:
- 提高置信度阈值
- 使用更强的NMS
- 增加负样本训练

### 场景3: 平衡的Precision和Recall

```
Precision: 0.78
Recall:    0.75
F1 Score:  0.765
```

**含义**:
- 模型性能良好且平衡
- 既不过于保守也不过于激进

**这是理想状态！**

---

## 常见问题 / FAQ

### Q1: mAP@0.5和mAP@0.5:0.95有什么区别?

**A**: 
- **mAP@0.5**: 只在IoU=0.5阈值下计算
- **mAP@0.5:0.95**: 在IoU从0.5到0.95 (步长0.05) 的多个阈值下计算，然后取平均

本实现目前支持mAP@0.5。要计算mAP@0.5:0.95，需要对每个IoU阈值重复计算AP。

### Q2: 为什么评估时conf_threshold设为0.001?

**A**: 
- 评估时使用很低的阈值 (如0.001)
- 这样可以保留所有预测
- AP计算会考虑所有置信度级别的性能
- 推理时可以根据需要调整阈值

### Q3: 如何提高mAP?

**A**: 
1. **增加训练数据**: 更多样本提升泛化能力
2. **数据增强**: Mosaic, MixUp等
3. **调整超参数**: 学习率、batch size
4. **训练更长时间**: 增加epochs
5. **模型改进**: 使用更深或更宽的网络
6. **损失权重调整**: 平衡box/cls/dfl损失

### Q4: 训练多久可以评估一次?

**A**: 
- 建议每10-20个epoch评估一次
- 过于频繁会减慢训练速度
- 太少则无法及时发现问题

### Q5: 如何解读per-class AP?

**A**: 
- 某些类别AP很低可能表示:
  - 该类别样本太少
  - 该类别特征不明显
  - 该类别与其他类别容易混淆
- 可以针对性地增加该类别的训练样本

---

## 性能基准 / Performance Benchmark

### 不同数据集规模的典型mAP

| 数据集大小 | 训练Epochs | 预期mAP@0.5 |
|-----------|-----------|-------------|
| 50-100张  | 100       | 0.3 - 0.5   |
| 100-500张 | 100       | 0.5 - 0.7   |
| 500-1000张| 100       | 0.6 - 0.8   |
| 1000+张   | 100       | 0.7 - 0.9   |

*注: 具体数值取决于数据质量和任务难度*

### 不同img_size的影响

| img_size | 速度  | mAP@0.5 (相对) |
|----------|-------|----------------|
| 320      | 最快  | 基准 - 0.05    |
| 416      | 快    | 基准           |
| 640      | 中等  | 基准 + 0.05    |
| 800      | 慢    | 基准 + 0.08    |

---

## 最佳实践 / Best Practices

### 1. 训练策略

```python
# 初期 (1-30 epoch): 关注loss下降
# 不需要频繁评估

# 中期 (30-70 epoch): 开始评估
eval_interval = 10

# 后期 (70-100 epoch): 频繁评估寻找最佳点
eval_interval = 5
```

### 2. 早停策略

```python
# 如果mAP连续N个评估周期不提升，考虑停止
patience = 3
no_improve_count = 0

if current_mAP <= best_mAP:
    no_improve_count += 1
    if no_improve_count >= patience:
        print("Early stopping!")
        break
```

### 3. 模型选择

```python
# 不要只看最后一个epoch的模型
# 使用mAP最高的模型: yolov11_best.pth
model.load_state_dict(torch.load('yolov11_best.pth'))
```

---

## 总结 / Summary

### 关键要点

1. **mAP@0.5**: 最重要的综合指标
2. **Precision/Recall**: 分析模型行为
3. **F1 Score**: 平衡性评估
4. **Per-class AP**: 找出问题类别
5. **定期评估**: 监控训练进度
6. **保存最佳**: 基于mAP选择模型

### 评估流程总结

```
加载数据 → 模型推理 → 收集预测/GT → 
计算IoU → 标记TP/FP → 计算PR曲线 → 
计算AP → 计算mAP → 可视化
```

---

**祝你取得好的评估结果！Good luck with your evaluation! 🎯**
