# Bug Fix: Classification Loss Explosion

## Problem

The classification loss was disproportionately large (~2100) compared to box loss (~2.4) and DFL loss (~17), making the total loss imbalanced.

## Root Cause

In the original implementation (`criterion/detect_loss.py` line 467):

```python
cls_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum
```

The issue was:
1. `self.bce(pred_scores, target_scores)` returns a tensor of shape `[B, A, C]`
   - B = batch size (e.g., 2)
   - A = number of anchors (8400)
   - C = number of classes (7)
   - Total elements: 2 × 8400 × 7 = 117,600

2. `.sum()` sums over all 117,600 elements

3. `target_scores_sum` is the sum of target scores (typically ~5-10 for a few objects)

4. Result: 117,600 elements summed / small divisor = HUGE loss

## Solution

Changed the normalization to divide by batch size and number of anchors:

```python
cls_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / (B * pred_scores.shape[1])
```

This properly averages the loss across all spatial positions and batch samples.

## Results

**Before Fix:**
```
Total Loss: 4240.09
Box Loss: 2.41
Cls Loss: 2100.70  ← TOO LARGE!
DFL Loss: 16.94
```

**After Fix:**
```
Total Loss: 43.91
Box Loss: 2.40
Cls Loss: 2.49  ← BALANCED! ✅
DFL Loss: 17.07
```

## Why This Matters

1. **Training Stability**: Imbalanced losses can cause gradient explosion or make one loss dominate training
2. **Loss Weighting**: The hyperparameters (box: 7.5, cls: 0.5, dfl: 1.5) are designed assuming balanced base losses
3. **Convergence**: Proper normalization ensures all loss components contribute appropriately to learning

## Technical Details

The normalization factor `(B * num_anchors)` represents:
- **B (batch size)**: Averages loss per image
- **num_anchors (8400)**: Averages loss per anchor point

This gives an average loss per anchor per image, which is then weighted by the hyperparameters and multiplied by B at the end (line 486) to get the final total loss for the batch.

The key insight is that BCE loss should be normalized by the number of predictions (B × A), not by the number of positive targets, to avoid explosion when there are many more predictions than targets.

## Verification

All tests pass with the fix:
- ✅ Forward pass (train/eval modes)
- ✅ Loss calculation with balanced components
- ✅ Backward pass for training
- ✅ Box decoding during inference

Commit: d4e5734
