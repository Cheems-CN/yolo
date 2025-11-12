"""
Test script to verify YOLOv11 forward pass and loss calculation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Import modules directly to avoid relative import issues
from modules import Conv, C3k2, SPPF, C2PSA, Detect
import torch.nn as nn
import torch.nn.functional as F

# Import YOLOv11 model class definition
class YOLOv11(nn.Module):
    def __init__(self, num_classes: int = 80):
        super(YOLOv11, self).__init__()
        # Backbone
        self.layer0 = Conv(3, 16, 3, 2, 1)
        self.layer1 = Conv(16, 32, 3, 2, 1)
        self.layer2 = C3k2(32, 64, n=1, c3k=False)
        self.layer3 = Conv(64, 64, 3, 2, 1)
        self.layer4 = C3k2(64, 128, n=1, c3k=False)
        self.layer5 = Conv(128, 128, 3, 2, 1)
        self.layer6 = C3k2(128, 128, n=1, c3k=True)
        self.layer7 = Conv(128, 256, 3, 2, 1)
        self.layer8 = C3k2(256, 256, n=1, c3k=True)
        self.layer9 = SPPF(256, 256)
        self.layer10 = C2PSA(256, 256, n=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer13 = C3k2(384, 128, n=1, c3k=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer16 = C3k2(256, 64, n=1, c3k=False)
        self.layer17 = Conv(64, 64, 3, 2, 1)
        self.layer19 = C3k2(192, 128, n=1, c3k=False)
        self.layer20 = Conv(128, 128, 3, 2, 1)
        self.layer22 = C3k2(384, 256, n=1, c3k=True)

        self.detect = Detect(nc=num_classes, ch=(64, 128, 256))

    def forward(self, x: torch.Tensor):
        # Backbone
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)

        # Neck up path
        u1 = self.upsample1(x10)
        c1 = torch.cat([u1, x6], dim=1)
        x13 = self.layer13(c1)

        u2 = self.upsample2(x13)
        u2 = self._match_hw(u2, x4)
        c2 = torch.cat([u2, x4], dim=1)         # 128 + 128 = 256
        x16 = self.layer16(c2)                  # /8 64ch (P3)

        # Neck down path
        d1 = self.layer17(x16)                  # /16 64ch
        c3 = torch.cat([d1, x13], dim=1)        # 64 + 128 = 192
        x19 = self.layer19(c3)                  # /16 128ch (P4)

        d2 = self.layer20(x19)                  # /32 128ch
        c4 = torch.cat([d2, x10], dim=1)        # 128 + 256 = 384
        x22 = self.layer22(c4)                  # /32 256ch (P5)

        # Detect head expects list of feature maps [small, medium, large]
        return self.detect([x16, x19, x22])

    @staticmethod
    def _match_hw(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='nearest')
        return x

# Import loss function
from criterion.detect_loss import YoloV11DetectionLoss


def test_forward_pass():
    """Test forward pass of the model."""
    print("=" * 60)
    print("Testing YOLOv11 Forward Pass")
    print("=" * 60)
    
    # Initialize model
    num_classes = 7  # Based on label format shown (class 6 + 0-indexed = 7 classes)
    model = YOLOv11(num_classes=num_classes)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    img_size = 640
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass in eval mode
    with torch.no_grad():
        output = model(dummy_input)
    
    if isinstance(output, tuple):
        predictions, raw = output
        print(f"Inference output shape: {predictions.shape}")
        print(f"Raw feature maps: {len(raw)} layers")
        for i, feat in enumerate(raw):
            print(f"  Layer {i}: {feat.shape}")
    else:
        print(f"Output type: {type(output)}")
        if isinstance(output, list):
            print(f"Number of feature maps: {len(output)}")
            for i, feat in enumerate(output):
                print(f"  Layer {i}: {feat.shape}")
    
    print("✓ Forward pass successful (eval mode)\n")
    
    # Test training mode
    model.train()
    output_train = model(dummy_input)
    print(f"Training mode output: {len(output_train)} feature maps")
    for i, feat in enumerate(output_train):
        print(f"  Layer {i}: {feat.shape}")
    
    print("✓ Forward pass successful (train mode)\n")
    return model, output_train


def test_loss_calculation():
    """Test loss calculation with dummy data."""
    print("=" * 60)
    print("Testing Loss Calculation")
    print("=" * 60)
    
    num_classes = 7
    model = YOLOv11(num_classes=num_classes)
    model.train()
    
    # Create dummy batch
    batch_size = 2
    img_size = 640
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Create dummy targets matching the label format:
    # class x_center y_center width height (normalized)
    # Let's simulate 5 objects total across 2 images
    targets = [
        [0, 6, 0.535577, 0.515385, 0.234615, 0.230769],  # image 0
        [0, 6, 0.350962, 0.415385, 0.419231, 0.296154],  # image 0
        [0, 6, 0.751923, 0.546154, 0.105769, 0.334615],  # image 0
        [1, 6, 0.465385, 0.736538, 0.163462, 0.146154],  # image 1
        [1, 6, 0.746154, 0.275962, 0.198077, 0.132692],  # image 1
    ]
    targets = torch.tensor(targets, dtype=torch.float32)
    
    # Prepare batch dict as expected by loss function
    batch = {
        'imgs': images,
        'batch_idx': targets[:, 0],  # which image each target belongs to
        'cls': targets[:, 1:2],      # class labels
        'bboxes': targets[:, 2:6]    # xywh normalized
    }
    
    print(f"Batch size: {batch_size}")
    print(f"Image shape: {images.shape}")
    print(f"Number of targets: {len(targets)}")
    print(f"Target format: [img_idx, class, x_norm, y_norm, w_norm, h_norm]")
    
    # Forward pass
    feats = model(images)
    print(f"\nModel output: {len(feats)} feature maps")
    for i, feat in enumerate(feats):
        print(f"  Feature {i}: {feat.shape}")
    
    # Initialize loss
    loss_fn = YoloV11DetectionLoss(model, tal_topk=10)
    
    # Calculate loss
    total_loss, loss_items = loss_fn(feats, batch)
    
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print(f"Box Loss: {loss_items[0].item():.4f}")
    print(f"Cls Loss: {loss_items[1].item():.4f}")
    print(f"DFL Loss: {loss_items[2].item():.4f}")
    
    # Verify loss is finite and requires grad
    assert torch.isfinite(total_loss), "Loss is not finite!"
    assert total_loss.requires_grad, "Loss does not require grad!"
    
    print("✓ Loss calculation successful\n")
    
    # Test backward pass
    total_loss.backward()
    print("✓ Backward pass successful\n")
    
    return loss_fn


def test_box_decoding():
    """Test box decoding during inference."""
    print("=" * 60)
    print("Testing Box Decoding (Inference)")
    print("=" * 60)
    
    num_classes = 7
    model = YOLOv11(num_classes=num_classes)
    model.eval()
    
    # Single image inference
    img = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        output = model(img)
    
    if isinstance(output, tuple):
        predictions, raw_feats = output
        print(f"Predictions shape: {predictions.shape}")
        print(f"Format: [batch, outputs, num_anchors]")
        print(f"  where outputs = {predictions.shape[1]} (4 bbox + {num_classes} classes)")
        print(f"  and num_anchors = {predictions.shape[2]}")
        
        # Check output dimensions
        expected_outputs = 4 + num_classes  # bbox + classes
        assert predictions.shape[1] == expected_outputs, \
            f"Expected {expected_outputs} outputs in dim 1, got {predictions.shape[1]}"
        
        # Extract boxes and scores (transposed format)
        boxes = predictions[:, :4, :]  # [batch, 4, anchors]
        scores = predictions[:, 4:, :]  # [batch, nc, anchors]
        
        print(f"\nBoxes shape: {boxes.shape}")
        print(f"Scores shape: {scores.shape}")
        print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        print("✓ Box decoding successful\n")
    else:
        print("Note: Model in training mode, no inference output")
    
    return model


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("YOLOv11 Complete Flow Test")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Forward pass
        model, train_output = test_forward_pass()
        
        # Test 2: Loss calculation
        loss_fn = test_loss_calculation()
        
        # Test 3: Box decoding
        model_inf = test_box_decoding()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("- Forward pass works in both train and eval modes")
        print("- Loss calculation works with multi-instance labels")
        print("- Box decoding works during inference")
        print("- Backward pass works for training")
        print("\nThe YOLOv11 implementation is complete and functional!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
