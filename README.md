# Vesuvius Challenge - Surface Detection

**Kaggle Competition**: Detect papyrus surfaces in 3D CT scans of ancient scrolls
**Prize Pool**: $100,000
**Deadline**: February 13, 2026
**Current Top Score**: 0.580

## 📊 Competition Overview

Segment papyrus sheet surfaces from 3D CT scans using:
- **TopoScore** (30%) - Topological correctness
- **SurfaceDice@τ=2.0** (35%) - Boundary accuracy
- **VOI** (35%) - Segmentation quality

## 📁 Repository Structure

```
├── compass_artifact_wf-*.md              # Competition research & strategy
├── best_patch_model.pth                  # Trained 3D U-Net checkpoint (22MB)
│
├── Advanced Kaggle Notebooks (485+ votes):
│   ├── inference-vesuvius-surface-3d-detection.ipynb
│   ├── surface-train-inference-3d-segm-gpu-augment.ipynb
│   └── train-transunet-baseline-lb-0-537.ipynb
│
├── Data:
│   ├── train_images/          # 807 volumes (320³ TIFF)
│   ├── train_labels/          # Binary segmentation masks
│   ├── test_images/           # Test volumes
│   ├── train.csv              # Training metadata
│   ├── train_clean.csv        # Cleaned training data
│   └── test.csv               # Test metadata
```

## 🏆 Key Techniques from Top Solutions (LB 0.545-0.580)

### 1. **Test Time Augmentation (TTA)** - +8% improvement
- 7 augmentations: original + 3 flips + 3 rotations
- Ensemble via mean logits

### 2. **Hysteresis Post-Processing**
- Two-threshold approach (T_low=0.5, T_high=0.9)
- Anisotropic morphological closing
- Small object removal (min_size=100)

### 3. **Skeleton-Aware Loss**
- Multi-component: Dice+CE + Skeleton Recall + FP suppression
- Forces model to learn thin tubular structures
- Weights: [1.0, 0.75, 0.5]

### 4. **TransUNet Architecture**
- SEResNeXt50 encoder
- Better than basic 3D U-Net
- Input: (160, 160, 160, 1)

### 5. **GPU-Accelerated Training**
- MONAI transforms
- Gradient accumulation (18 steps)
- Mixed precision (FP16)
- Tversky Loss (α=0.7, β=0.3)

## 🚀 Next Steps

1. **Architecture**: Implement TransUNet + SEResNeXt50
2. **Loss**: Add skeleton-aware multi-component loss
3. **Augmentation**: 3D random occlusions + GPU transforms
4. **Inference**: TTA with 7 augmentations
5. **Post-processing**: Hysteresis thresholding

## 📚 References

- [Kaggle Competition](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
- [Top Inference Notebook (485 votes)](https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection)
- [GPU Augmentation Training (313 votes)](https://www.kaggle.com/code/jirkaborovec/surface-train-inference-3d-segm-gpu-augment)
- [TransUNet Baseline (LB 0.537)](https://www.kaggle.com/code/choudharymanas/train-transunet-baseline-lb-0-537)
- [ThaumatoAnakalyptor (Grand Prize Winner)](https://github.com/schillij95/ThaumatoAnakalyptor)
- [Vesuvius Ink Detection Winner](https://github.com/ainatersol/Vesuvius-InkDetection)

## 🛠️ Requirements

```bash
pip install torch torchvision tifffile scipy scikit-image
pip install monai  # For advanced augmentations
pip install kaggle  # For submissions
```

## 📝 Current Status

- ✅ Downloaded top 3 Kaggle notebooks
- ✅ Analyzed winning techniques
- ✅ Trained basic 3D U-Net (Dice loss: 0.2843, incomplete)
- ⏳ Next: Implement TransUNet + advanced techniques
- 🎯 Target: LB > 0.55
