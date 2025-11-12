# 故障排查指南 / Troubleshooting Guide

## 问题：推理时所有边界框坐标为0 / Issue: All Bounding Box Coordinates are 0 During Inference

### 症状 / Symptoms

运行`inference_example.py`时，输出显示：
```
检测到 8 个目标 / Detected 8 objects
  目标 1: 位置=(x=0.0, y=0.0, w=0.0, h=0.0)
  目标 2: 位置=(x=0.0, y=0.0, w=0.0, h=0.0)
  ...
```

图片上没有绘制任何检测框。

### 可能原因 / Possible Causes

#### 1. ⚠️ 模型未训练 (最常见 / Most Common)

**问题**: 使用的是未训练的模型权重

**检查方法**:
```python
# 查看模型文件大小
# Check model file size
import os
print(f"模型大小: {os.path.getsize('yolov11_final.pth') / 1024 / 1024:.2f} MB")

# 应该 > 5MB，如果只有几KB说明是空模型
# Should be > 5MB, if only a few KB it's an empty model
```

**解决方法**:
1. 先运行 `python train_example.py` 训练模型
2. 至少训练10-20个epoch
3. 确认loss下降到合理范围 (Total Loss < 50)
4. 训练完成后会生成 `yolov11_final.pth`

#### 2. 🔧 img_size不匹配

**问题**: 推理时使用的`img_size`与训练时不同

**示例**:
- 训练时: `img_size = 640`
- 推理时: `img_size = 312` ← 错误！

**检查方法**:
```python
# 在 train_example.py 中
img_size = 640  # 记住这个值

# 在 inference_example.py 中
img_size = 640  # 必须相同！
```

**解决方法**:
- 确保 `train_example.py` 和 `inference_example.py` 中的 `img_size` **完全相同**
- 如果训练时用的是320，推理时也必须用320

#### 3. 📦 权重文件路径错误

**问题**: 模型文件不存在或路径错误

**检查方法**:
```python
import os
model_path = "yolov11_final.pth"
if not os.path.exists(model_path):
    print(f"❌ 文件不存在: {model_path}")
else:
    print(f"✓ 文件存在: {model_path}")
```

**解决方法**:
- 确认权重文件存在于正确的路径
- 使用绝对路径: `model_path = r"D:\yolo_git\yolov11_final.pth"`

#### 4. 🎯 置信度阈值过高

**问题**: `conf_threshold` 设置太高，过滤掉了所有检测

**检查方法**:
```python
# 在 inference_example.py 中临时降低阈值
conf_threshold = 0.01  # 尝试很低的值
```

**解决方法**:
- 对于未充分训练的模型，使用较低的阈值: `0.01` - `0.15`
- 训练好的模型可以使用: `0.25` - `0.5`

#### 5. 🔢 类别数量不匹配

**问题**: 训练时的`num_classes`与推理时不同

**检查方法**:
```python
# 训练时 (train_example.py)
num_classes = 7  # 记住这个值

# 推理时 (inference_example.py)
num_classes = 7  # 必须相同
```

**解决方法**:
- 确保两处的`num_classes`一致
- 必须与数据集的实际类别数相同

---

## 问题：训练时Loss不下降 / Issue: Loss Not Decreasing During Training

### 症状 / Symptoms

```
Epoch [1] Loss: 5000.0
Epoch [2] Loss: 5000.0
Epoch [3] Loss: 5000.0
...
```

### 解决方案 / Solutions

#### 1. 检查数据标签 / Check Data Labels

```python
# 验证标签格式
# Verify label format
with open('dataset/labels/img1.txt') as f:
    for line in f:
        parts = line.strip().split()
        assert len(parts) == 5, "标签格式错误！应该是：class x y w h"
        cls, x, y, w, h = map(float, parts)
        assert 0 <= x <= 1 and 0 <= y <= 1, "坐标必须归一化到[0,1]"
        assert 0 < w <= 1 and 0 < h <= 1, "宽高必须归一化到[0,1]"
        print(f"✓ 标签正确: class={cls}, x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
```

#### 2. 降低学习率 / Reduce Learning Rate

```python
# 在 train_example.py 中
learning_rate = 0.0001  # 从0.001降到0.0001
```

#### 3. 检查数据集大小 / Check Dataset Size

```python
dataset = YOLODataset(image_dir, label_dir, img_size)
print(f"数据集大小: {len(dataset)} 张图片")

# 建议: 至少50-100张图片用于训练
# Recommendation: At least 50-100 images for training
```

#### 4. 使用更小的batch_size / Use Smaller Batch Size

```python
# 在 train_example.py 中
batch_size = 2  # 从4或8降到2
```

---

## 问题：CUDA内存溢出 / Issue: CUDA Out of Memory

### 症状 / Symptoms

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

### 解决方案 / Solutions

#### 1. 减小batch_size

```python
# train_example.py
batch_size = 2  # 或 1
```

#### 2. 减小img_size

```python
# train_example.py 和 inference_example.py
img_size = 416  # 从640降到416或320
```

#### 3. 使用CPU训练 (慢但稳定)

```python
# train_example.py
device = torch.device('cpu')  # 强制使用CPU
```

---

## 问题：图片大小警告 / Issue: Image Size Warnings

### 症状 / Symptoms

```
原始图片尺寸: (52, 52)
输入图片尺寸: 312x312
预测输出形状: torch.Size([1, 11, 2021])  # 锚点数不是8400
```

### 说明 / Explanation

这是**正常的**！不同的`img_size`会产生不同数量的锚点：

| img_size | 锚点数量 | 说明 |
|----------|---------|------|
| 320×320  | 3360    | P3:40×40, P4:20×20, P5:10×10 |
| 416×416  | 5460    | P3:52×52, P4:26×26, P5:13×13 |
| 640×640  | 8400    | P3:80×80, P4:40×40, P5:20×20 |

代码会自动处理不同数量的锚点，**无需担心**。

### 注意事项 / Important Note

⚠️ **必须保证训练和推理使用相同的`img_size`！**
- 训练用640 → 推理也用640
- 训练用416 → 推理也用416

---

## 调试技巧 / Debugging Tips

### 1. 打印模型预测原始值

```python
# 在 inference_example.py 的 main() 函数中添加
with torch.no_grad():
    predictions, raw_features = model(image_tensor)

# 打印统计信息
print(f"预测值范围:")
print(f"  最小值: {predictions.min().item():.4f}")
print(f"  最大值: {predictions.max().item():.4f}")
print(f"  平均值: {predictions.mean().item():.4f}")

# 打印前4个通道(boxes)的值
boxes = predictions[0, :4, :]
print(f"\nBox通道统计:")
print(f"  X范围: [{boxes[0].min():.2f}, {boxes[0].max():.2f}]")
print(f"  Y范围: [{boxes[1].min():.2f}, {boxes[1].max():.2f}]")
print(f"  W范围: [{boxes[2].min():.2f}, {boxes[2].max():.2f}]")
print(f"  H范围: [{boxes[3].min():.2f}, {boxes[3].max():.2f}]")
```

**正常输出应该是**:
- Boxes的值在0到img_size之间 (如0-640)
- 不应该全是0或全是相同的值

### 2. 验证模型是否正确加载

```python
# 在加载模型后
model_dict = torch.load(model_path, map_location=device)
print("模型权重键名:")
for i, key in enumerate(list(model_dict.keys())[:5]):
    print(f"  {i+1}. {key}: {model_dict[key].shape}")

# 检查是否有非零值
has_nonzero = any((v != 0).any() for v in model_dict.values())
print(f"模型包含非零权重: {has_nonzero}")
```

### 3. 测试单个图片

```python
# 创建简单测试脚本
import torch
from yolov11 import YOLOv11

model = YOLOv11(num_classes=7)
model.load_state_dict(torch.load('yolov11_final.pth'))
model.eval()

# 随机输入
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    out, _ = model(x)

print(f"输出形状: {out.shape}")
print(f"输出范围: [{out.min():.2f}, {out.max():.2f}]")
print(f"包含NaN: {torch.isnan(out).any()}")
print(f"包含Inf: {torch.isinf(out).any()}")
```

---

## 快速检查清单 / Quick Checklist

推理前检查 / Before Inference:

- [ ] 模型已训练完成 (至少10+ epochs)
- [ ] 训练loss已下降到合理范围 (< 100)
- [ ] `yolov11_final.pth` 文件存在且 > 5MB
- [ ] `img_size` 在训练和推理脚本中**完全相同**
- [ ] `num_classes` 在训练和推理脚本中**完全相同**
- [ ] 图片路径正确
- [ ] 置信度阈值合理 (0.1 - 0.5)

训练前检查 / Before Training:

- [ ] 数据集目录结构正确 (dataset/images/, dataset/labels/)
- [ ] 每个PNG图片都有对应的TXT标签文件
- [ ] 标签格式正确: `class x y w h` (归一化)
- [ ] 类别数量设置正确
- [ ] GPU显存足够 (或使用CPU)
- [ ] batch_size和img_size根据显存调整

---

## 获取帮助 / Getting Help

如果以上方法都无法解决问题：

1. **收集信息**:
   ```bash
   # 运行诊断脚本
   python -c "
   import torch
   import sys
   print(f'Python版本: {sys.version}')
   print(f'PyTorch版本: {torch.__version__}')
   print(f'CUDA可用: {torch.cuda.is_available()}')
   if torch.cuda.is_available():
       print(f'CUDA版本: {torch.version.cuda}')
       print(f'GPU: {torch.cuda.get_device_name(0)}')
   "
   ```

2. **提供以下信息**:
   - 完整的错误信息
   - 训练和推理脚本中的关键参数 (img_size, num_classes, batch_size)
   - 数据集大小和图片尺寸
   - 训练了多少个epoch
   - 最终的loss值

3. **常见解决方案优先级**:
   1. 确认模型已训练 ⭐⭐⭐⭐⭐
   2. 检查img_size一致性 ⭐⭐⭐⭐
   3. 降低conf_threshold ⭐⭐⭐
   4. 验证数据标签格式 ⭐⭐⭐
   5. 重新训练模型 ⭐⭐

---

**祝你成功！Good luck! 🚀**
