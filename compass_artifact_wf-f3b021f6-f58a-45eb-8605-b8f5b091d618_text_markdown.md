# Optimal solutions for Vesuvius Challenge Surface Detection

**The key to winning this topology-focused competition lies in combining proven 3D segmentation architectures with explicit topological constraints and careful post-processing.** Based on solutions from prior Vesuvius competitions and state-of-the-art research, the optimal approach merges nnU-Net-style training pipelines with topology-aware loss functions like clDice, followed by connected component post-processing to eliminate mergers and splits. The competition's unique evaluation combining TopoScore (30%), SurfaceDice@τ=2.0 (35%), and VOI (35%) demands attention to both voxel accuracy and structural connectivity—a balance that requires explicit topological supervision during training.

## Competition context and winning precedents

The Vesuvius Challenge Surface Detection competition runs until **February 13, 2026** with a $100,000 prize pool. The task requires detecting papyrus surfaces inside 3D CT scans of ancient Herculaneum scrolls, with outputs feeding the virtual unwrapping pipeline for reading these carbonized manuscripts. Unlike typical medical segmentation, this domain presents unique challenges: scroll sheets are extremely thin, densely wrapped, and the semi-automatically generated labels may contain topological artifacts.

Prior Vesuvius competitions reveal crucial insights. The **2023 Ink Detection first-place solution** (Team ryches) used a two-stage architecture: 3D CNNs/UNets/UNETR models outputting multi-channel volumes, followed by **SegFormer 2D segmentation** that proved depth-invariant. Their ensemble of 9 models achieved F0.5 = 0.683. The **$700,000 Grand Prize winner** (Youssef Nader's team) emphasized domain adaptation and discovered that smaller **64×64 windows outperformed larger contexts**—larger windows biased predictions toward fixed brush widths while smaller windows produced more reliable, interpretable results.

Julian Schilliger's **ThaumatoAnakalyptor** pipeline demonstrates a classical-plus-deep-learning approach: 3D Sobel kernels for surface detection, PointCloud generation with local normal calculation, **Mask3D** for instance segmentation, and graph-based stitching with winding angle optimization. This hybrid methodology—combining gradient-based edge detection with learned refinement—may prove valuable for the topology-constrained Surface Detection task.

## Architecture selection: nnU-Net remains the gold standard

A critical **MICCAI 2024 paper** ("nnU-Net Revisited: A Call for Rigorous Validation") demonstrated that when properly evaluated, CNN-based U-Net models using the nnU-Net framework still match or exceed Transformer and Mamba-based alternatives. Claims of architectural superiority from newer methods often stemmed from inadequate baselines rather than genuine improvements.

**nnU-Net's self-configuring pipeline** automatically handles:
- **Fixed parameters**: Dice + Cross-Entropy loss, SGD with momentum 0.99, polynomial learning rate decay, 1000 epochs
- **Rule-based parameters**: Network topology adapted to patch size, jointly optimized batch size and depth for GPU memory, z-score normalization for CT data
- **Empirical parameters**: Selection between 2D, 3D fullres, and cascade configurations via 5-fold cross-validation

For the Vesuvius challenge, the **3d_fullres configuration** with **Residual Encoder (ResEnc) presets** provides the strongest baseline. The cascade approach (low-resolution global model → high-resolution refinement) handles large volumes exceeding GPU memory.

**Swin UNETR** offers an alternative when global context matters—its shifted-window attention captures long-range dependencies better than CNNs. MONAI provides pretrained weights from 5,050 CT images using self-supervised learning. However, transformers require more training data without pretraining and are sensitive to hyperparameters.

For efficiency-constrained Kaggle notebooks, emerging **SegMamba** architectures offer O(n) complexity versus Transformer's O(n²), maintaining accuracy while reducing computational cost. The Tri-orientated Spatial Mamba (ToM) module processes features across axial, sagittal, and coronal planes simultaneously.

## Topology-preserving training strategies

The competition's TopoScore metric uses **Betti matching**—a method from persistent homology that matches topological features spatially, not just counting them. Traditional Betti number error can mark two features as matching even if located in entirely different positions; Betti matching requires spatial correspondence, making it far more sensitive to actual topological errors.

Three key topological loss functions show promise:

**clDice (Centerline Dice)** computes Dice on the intersection of segmentation masks with their morphological skeleta. When clDice = 1, topology preservation is theoretically guaranteed up to homotopy equivalence. The soft-clDice implementation uses iterative soft-skeletonization via max-pooling operations—fully differentiable with standard deep learning functions. Combined loss: `L = α × L_soft-clDice + (1-α) × L_soft-Dice` provides excellent topology-accuracy trade-offs.

**TopoLoss** uses Wasserstein distance between persistence diagrams of predictions and ground truth, identifying critical pixels where topological changes occur and reweighting them during training. Total loss: `L = L_bce + λ × L_topo`. When L_topo = 0, segmentation is mathematically guaranteed to have the same Betti number as ground truth.

**DMT-Loss (Discrete Morse Theory)** identifies global structures—1D stable manifolds (skeleton-like) and 2D stable manifolds (sheet-like)—then applies cross-entropy constrained to these topologically critical regions. This converges faster than pure persistent homology approaches and captures extended structures rather than just critical points.

For preventing specific errors:
- **Mergers**: PH-based loss penalizes when predicted β₀ < ground truth β₀; DMT 2-stable manifolds identify separating sheets
- **Splits**: clDice's Tsens term enforces skeleton connectivity; warping error explicitly counts split errors  
- **Spurious holes/handles**: Topological priors specifying expected Betti numbers; PH loss with β₁ constraints

## Optimizing for VOI and Surface Dice metrics

The **Variation of Information (VOI)** metric decomposes into two components: VOI_split (over-segmentation, H(X|Y)) and VOI_merge (under-segmentation, H(Y|X)). Perfect over-segmentation yields H(X|Y) = 0 but high VOI_merge; perfect under-segmentation yields H(Y|X) = 0 but high VOI_split. The competition likely reports both separately, so balancing this trade-off is crucial.

**Boundary-aware loss functions** correlate strongly with VOI:
- **Boundary Loss** uses distance transforms multiplied with softmax outputs, complementing regional losses for unbalanced segmentation
- **BALoss** detects "leakage" pixels at boundaries, achieving ~15% better VOI scores on CREMI/ISBI benchmarks
- **Affinity prediction** with watershed and agglomeration naturally optimizes for VOI through learned merge functions

**Surface Dice at τ=2.0** measures boundary accuracy within a 2-pixel tolerance. This requires explicit attention to contour quality beyond volumetric overlap. Combining boundary losses with standard Dice during training typically improves this metric.

For VOI optimization, the **Local Shape Descriptors (LSD)** approach from Nature Methods 2022 adds an auxiliary learning task predicting local shape statistics. ACLSD/ACRLSD auto-context networks achieve VOI scores competitive with Flood-Filling Networks at 100× lower computational cost.

## Practical implementation for Kaggle constraints

The 9-hour notebook runtime and 16GB GPU memory require careful optimization. **Mixed precision training** reduces memory by ~50% and provides 1.5-5.5× speedup:

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)
    loss = loss_fn(output, target)
scaler.scale(loss).backward()
```

**Patch-based training** with class-balanced sampling handles variable-sized volumes. nnU-Net automatically determines optimal patch size based on median image size, target spacing, and GPU constraints. For inference, **sliding window with Gaussian weighting** blends overlapping predictions—50% overlap is typical, though 25% trades accuracy for speed.

Critical **data augmentation** for scroll CT data includes:
- Rotation up to 180° (critical for scroll orientation invariance)
- Axis flipping (all three axes)
- Elastic deformation (simulates realistic tissue warping)
- Intensity normalization: clip to [0.5, 99.5] percentiles, then z-score normalize using foreground statistics

**Post-processing** finalizes topological correctness:
- Connected component analysis with 26-connectivity
- Remove components below volume threshold
- Morphological closing fills small holes
- Gaussian importance weighting during inference reduces stitching artifacts

**Test-time augmentation** with 5-8 flip combinations improves Dice by 0.1-2.3% but multiplies inference time proportionally—profile carefully against the runtime limit.

## Domain-specific considerations for carbonized papyrus

Scroll CT scans present unique challenges absent from typical medical imaging. The papyrus sheets are **extremely thin** (often just a few voxels thick), densely wrapped in tight spirals with minimal separation between layers. The carbonization process created nearly uniform density throughout the scroll, making sheet boundaries difficult to distinguish.

The **semi-automatically labeled training data** likely contains topological artifacts—this is explicitly acknowledged in the competition description. Training with topology-aware losses helps the model learn correct topology despite imperfect supervision. Consider using **label smoothing** or **soft labels** near boundaries where annotation uncertainty is highest.

**Z-translation invariance** proved critical in prior Vesuvius competitions—the 4th place Kaggle solution identified this as a major performance booster. Implementing z-axis shift augmentation and ensuring the network doesn't memorize absolute depth positions improves generalization.

The recto surface detection requirement (horizontal fibers facing umbilicus) suggests the model must learn subtle fiber orientation patterns, not just sheet location. Including **orientation-aware features** or explicitly predicting local fiber direction as an auxiliary task may help.

## Recommended solution architecture

Based on this research, the optimal approach combines:

1. **Base architecture**: nnU-Net 3d_fullres with ResEnc presets, or Swin UNETR if pretrained weights transfer well
2. **Loss function**: Dice + Cross-Entropy + λ×clDice (or TopoLoss for explicit topological supervision)
3. **Training**: 5-fold cross-validation, 1000 epochs, class-balanced patch sampling, heavy augmentation including 180° rotation and z-translation
4. **Post-processing**: Connected component analysis, morphological closing, small fragment removal
5. **Inference**: Sliding window with 50% overlap and Gaussian weighting, conservative TTA (5-8 augmentations)

For pushing beyond baselines, consider:
- **Hybrid 3D-2D architecture** like the winning Ink Detection solution (3D encoder → 2D SegFormer decoder)
- **Auto-context networks** that iteratively refine predictions using prior segmentation as additional input
- **Instance segmentation** via Mask3D if the evaluation distinguishes between individual sheet instances
- **Affinity-based methods** with learned agglomeration if VOI optimization proves challenging

The competition's emphasis on topological correctness makes explicit topology-aware training essential—teams relying solely on pixel-wise losses will likely struggle with the TopoScore and VOI components despite achieving good voxel accuracy.