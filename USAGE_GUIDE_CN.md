# YOLOv11 使用指南 / Usage Guide

## 目录 / Table of Contents
1. [数据准备](#数据准备)
2. [训练模型](#训练模型)
3. [推理预测](#推理预测)
4. [测试说明](#测试说明)

---

## 数据准备

### 数据集目录结构

请按以下结构组织你的数据集：

```
your_project/
├── dataset/
│   ├── images/          # PNG图片目录
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   └── labels/          # TXT标签目录
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
├── yolov11.py
├── train_example.py
└── ...
```

### 标签格式

每个PNG图片对应一个同名的TXT标签文件，格式如下：

```
class_id x_center y_center width height
```

**重要说明：**
- `class_id`: 类别ID，从0开始的整数（例如：0, 1, 2, ..., 6）
- `x_center`: 边界框中心的x坐标，**归一化到[0,1]**（相对于图片宽度）
- `y_center`: 边界框中心的y坐标，**归一化到[0,1]**（相对于图片高度）
- `width`: 边界框的宽度，**归一化到[0,1]**（相对于图片宽度）
- `height`: 边界框的高度，**归一化到[0,1]**（相对于图片高度）

**示例标签文件** (img1.txt):
```
6 0.535577 0.515385 0.234615 0.230769
6 0.350962 0.415385 0.419231 0.296154
6 0.751923 0.546154 0.105769 0.334615
```

这个例子表示图片中有3个目标，都属于类别6。

### 坐标归一化示例

假设你的原始图片大小是 800×600 像素，一个目标的边界框坐标是：
- 中心点: (400, 300)
- 宽度: 200
- 高度: 150

那么归一化后的坐标应该是：
```
x_center = 400 / 800 = 0.5
y_center = 300 / 600 = 0.5
width = 200 / 800 = 0.25
height = 150 / 600 = 0.25
```

标签文件中写：
```
0 0.5 0.5 0.25 0.25
```
（假设这个目标是类别0）

---

## 训练模型

### 1. 修改训练脚本配置

编辑 `train_example.py` 文件，修改以下配置：

```python
# 数据集路径
image_dir = "dataset/images"  # 改为你的图片目录
label_dir = "dataset/labels"  # 改为你的标签目录

# 训练参数
num_classes = 7      # 改为你的类别数量
img_size = 640       # 图片大小
batch_size = 4       # 批次大小（根据显存调整）
num_epochs = 100     # 训练轮数
learning_rate = 0.001  # 学习率
```

### 2. 运行训练

```bash
python train_example.py
```

### 3. 训练输出

训练过程中会显示：
```
Epoch [1] Batch [10/100] Loss: 45.3421 (Box: 2.41, Cls: 2.53, DFL: 17.06)
Epoch [1] Batch [20/100] Loss: 43.8762 (Box: 2.38, Cls: 2.49, DFL: 16.98)
...
Epoch [1] 平均损失 / Average Loss: 44.1234 (Box: 2.40, Cls: 2.51, DFL: 17.02)
```

**损失说明：**
- **Total Loss**: 总损失（Box + Cls + DFL）
- **Box Loss**: 边界框回归损失（CIoU）
- **Cls Loss**: 分类损失（BCE）
- **DFL Loss**: 分布焦点损失（用于精确定位）

### 4. 模型保存

训练完成后，模型会保存到：
- `checkpoints/yolov11_epoch_10.pth` (每10个epoch保存一次)
- `checkpoints/yolov11_epoch_20.pth`
- ...
- `yolov11_final.pth` (最终模型)

---

## 推理预测

### 1. 修改推理脚本配置

编辑 `inference_example.py` 文件，修改以下配置：

```python
model_path = "yolov11_final.pth"  # 训练好的模型路径
image_path = "test_image.png"     # 要预测的图片路径
num_classes = 7                    # 类别数量（与训练时一致）
conf_threshold = 0.25              # 置信度阈值
iou_threshold = 0.45               # NMS的IoU阈值

# 自定义类别名称（可选）
class_names = ["defect_1", "defect_2", "defect_3", ...]
```

### 2. 运行推理

```bash
python inference_example.py
```

### 3. 推理输出

```
使用设备 / Using device: cuda
加载模型 / Loading model...
模型已加载 / Model loaded from: yolov11_final.pth

加载图片 / Loading image: test_image.png
原始图片尺寸 / Original image size: (1024, 768)
输入图片尺寸 / Input image size: 640x640

开始推理 / Starting inference...
预测输出形状 / Prediction output shape: torch.Size([1, 11, 8400])

后处理预测结果 / Post-processing predictions...
检测到 5 个目标 / Detected 5 objects
  目标 1 / Object 1: 类别 / Class=6, 置信度 / Confidence=0.892, 位置 / Position=(x=548.3, y=395.2, w=240.1, h=177.4)
  目标 2 / Object 2: 类别 / Class=6, 置信度 / Confidence=0.856, 位置 / Position=(x=359.4, y=318.7, w=429.3, h=227.5)
  ...

可视化结果 / Visualizing results...
结果已保存到 / Result saved to: detection_result.png

推理完成！/ Inference complete!
```

检测结果会保存为 `detection_result.png`，包含带标注的边界框和置信度。

---

## 测试说明

### 为什么测试脚本不需要数据集？

`test_yolo_flow.py` 是一个**单元测试脚本**，用于验证模型架构和损失函数的正确性。它使用**随机生成的假数据**来测试：

1. **前向传播**：模型能否正常处理输入
2. **损失计算**：损失函数能否正确计算
3. **反向传播**：梯度能否正常回传
4. **推理模式**：模型能否输出正确格式的预测

### 测试脚本中的假数据

```python
# 生成随机图片
images = torch.randn(batch_size, 3, img_size, img_size)

# 生成假标签（使用你提供的格式示例）
targets = [
    [0, 6, 0.535577, 0.515385, 0.234615, 0.230769],  # 图片0的目标
    [0, 6, 0.350962, 0.415385, 0.419231, 0.296154],  # 图片0的目标
    [0, 6, 0.751923, 0.546154, 0.105769, 0.334615],  # 图片0的目标
    [1, 6, 0.465385, 0.736538, 0.163462, 0.146154],  # 图片1的目标
    [1, 6, 0.746154, 0.275962, 0.198077, 0.132692],  # 图片1的目标
]
```

这些数据只是用来**验证代码能否正常运行**，不是用来训练模型的。

### 如何运行测试

```bash
python test_yolo_flow.py
```

如果看到 `✓ ALL TESTS PASSED!`，说明代码实现正确，可以开始使用你的真实数据集进行训练。

---

## 常见问题 FAQ

### Q1: 类别数量如何确定？

A: 统计你的数据集中有多少种不同的目标类别。例如，如果你的标签文件中类别ID有 0, 1, 2, 3, 4, 5, 6，那么 `num_classes = 7`。

### Q2: 批次大小如何设置？

A: 根据你的GPU显存调整：
- 4GB显存: batch_size = 2-4
- 8GB显存: batch_size = 4-8
- 16GB显存: batch_size = 8-16

如果出现 "CUDA out of memory" 错误，减小 batch_size。

### Q3: 训练多少个epoch？

A: 建议：
- 小数据集（<1000张图）: 100-200 epochs
- 中数据集（1000-5000张图）: 50-100 epochs
- 大数据集（>5000张图）: 30-50 epochs

观察损失曲线，如果损失不再下降，可以提前停止。

### Q4: 损失值多少算正常？

A: 初始损失：
- Total Loss: 40-50
- Box Loss: 2-3
- Cls Loss: 2-3
- DFL Loss: 15-20

训练后损失会逐渐下降，但具体数值取决于数据集难度。

### Q5: 如何提高检测精度？

A:
1. **增加训练数据**：更多的标注数据
2. **数据增强**：旋转、翻转、缩放等
3. **调整学习率**：尝试 0.0001 或 0.01
4. **训练更长时间**：增加 epochs
5. **调整超参数**：conf_threshold, iou_threshold

### Q6: 推理速度慢怎么办？

A:
1. 使用GPU推理（`device='cuda'`）
2. 减小输入图片尺寸（例如从640改为416）
3. 批量推理多张图片

---

## 完整工作流程示例

```bash
# 1. 准备数据集
mkdir -p dataset/images dataset/labels
# 将PNG图片放到 dataset/images/
# 将TXT标签放到 dataset/labels/

# 2. 运行测试（验证代码）
python test_yolo_flow.py

# 3. 训练模型
python train_example.py

# 4. 推理预测
python inference_example.py
```

---

## 技术支持

如有问题，请参考：
- `README.md`: 项目概述和架构说明
- `IMPLEMENTATION_NOTES.md`: 技术实现细节
- `BUGFIX_CLS_LOSS.md`: 分类损失修复说明

或在GitHub仓库提issue。
