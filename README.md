# YOLOv11 Implementation

A minimal and streamlined implementation of YOLOv11 object detection framework.

## Features

- ✅ Complete YOLOv11 architecture with backbone and detection head
- ✅ Task-Aligned Assigner for label assignment
- ✅ Distribution Focal Loss (DFL) for precise localization
- ✅ CIoU loss for bounding box regression
- ✅ Multi-instance label support
- ✅ Training and inference modes

## Architecture

### Model Components

- **Backbone**: Conv, C3k2 blocks with SPPF and C2PSA attention
- **Neck**: Feature Pyramid Network (FPN) with upsampling and concatenation
- **Head**: Detection head with box regression and classification branches

### Loss Function

The loss consists of three components:
1. **Box Loss (CIoU)**: Complete IoU loss for bounding box regression
2. **Classification Loss**: Binary Cross-Entropy with Logits
3. **DFL Loss**: Distribution Focal Loss for precise localization

## Usage

### 1. Model Initialization

```python
from yolov11 import YOLOv11

# Initialize model with number of classes
num_classes = 7  # Adjust based on your dataset
model = YOLOv11(num_classes=num_classes)
```

### 2. Prepare Data

Labels should be in the format: `class x_center y_center width height` (normalized to [0, 1])

Example label file:
```
6 0.535577 0.515385 0.234615 0.230769
6 0.350962 0.415385 0.419231 0.296154
6 0.751923 0.546154 0.105769 0.334615
```

### 3. Training

```python
import torch
from criterion.detect_loss import YoloV11DetectionLoss

# Set model to training mode
model.train()

# Initialize loss function
loss_fn = YoloV11DetectionLoss(model, tal_topk=10)

# Prepare batch
batch = {
    'imgs': images,           # [B, 3, H, W] tensor
    'batch_idx': batch_idx,   # [N] which image each target belongs to
    'cls': class_labels,      # [N, 1] class labels
    'bboxes': bboxes          # [N, 4] normalized xywh coordinates
}

# Forward pass
feats = model(images)

# Calculate loss
total_loss, (box_loss, cls_loss, dfl_loss) = loss_fn(feats, batch)

# Backward pass
total_loss.backward()
```

### 4. Inference

```python
# Set model to evaluation mode
model.eval()

# Single image inference
with torch.no_grad():
    predictions, raw_features = model(image)

# predictions shape: [batch, 4+num_classes, num_anchors]
# where:
#   - First 4 channels: bounding box coordinates (x, y, w, h) in pixels
#   - Next num_classes channels: class probabilities (sigmoid activated)
```

## Data Format

### Input Images
- Shape: `[batch_size, 3, height, width]`
- Typical size: 640x640 (can be adjusted)
- RGB format, normalized to [0, 1]

### Labels (Training)
Each row in label file:
```
class_id x_center y_center width height
```
All coordinates normalized to [0, 1]:
- `class_id`: integer class index (0-indexed)
- `x_center, y_center`: center of bounding box
- `width, height`: size of bounding box

### Output (Inference)
- Shape: `[batch, 4+num_classes, num_anchors]`
- Format: `[x, y, w, h, class_0_prob, class_1_prob, ..., class_n_prob]`
- Coordinates in pixels (multiply by stride and anchor position)
- Class probabilities after sigmoid activation

## Model Architecture Details

### Stride Levels
- P3: stride 8 (80x80 for 640x640 input)
- P4: stride 16 (40x40 for 640x640 input)
- P5: stride 32 (20x20 for 640x640 input)
- Total anchors: 8400 (6400 + 1600 + 400)

### Detection Head
- **reg_max**: 16 (DFL bins for each side of bbox)
- **Outputs per anchor**: 4*16 + num_classes
  - 4*16 = 64 for bbox (4 sides × 16 bins)
  - num_classes for classification

## Testing

Run the test suite to verify the implementation:

```bash
python test_yolo_flow.py
```

This will test:
- Forward pass in train and eval modes
- Loss calculation with multi-instance labels
- Box decoding during inference
- Backward pass for training

## Files

- `yolov11.py`: Main model architecture
- `modules/`: Model building blocks (Conv, C3k2, SPPF, C2PSA, Detect head)
- `criterion/detect_loss.py`: Loss function implementation
- `test_yolo_flow.py`: Comprehensive test suite

## Notes

- The implementation follows YOLOv11's Task-Aligned Assigner approach
- Supports multi-instance per image (multiple objects in one image)
- Minimal and streamlined code for easy understanding and modification
- All coordinates are normalized during training, converted to pixels during inference
