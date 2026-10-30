# Deep Learning Module Assessment: Technical Report
## Pixel-Level Road Anomaly Detection for Autonomous Driving

**Author:** Deep Learning Student Group  
**Dataset:** RoadAnomaly Benchmark (*Lis, Nakka, Fua, Salzmann - ICCV 2019*)  
**Submission Date:** October 2026 (Assessment Deadline: November 8, 2026)  
**Implementation Repository:** `RoadAnomaly_Project.ipynb`, `dataset.py`, `losses.py`, `models.py`, `train_and_evaluate.py`  

---

### Abstract
Autonomous driving safety relies on the perception system's ability to identify rare and unexpected hazards in the driving path, such as wild animals, road construction cones, lost tires, boulders, and non-standard vehicles. This paper presents an empirical investigation into deep learning approaches for pixel-level road anomaly detection on the **RoadAnomaly** dataset. We systematically explore the dataset's characteristics, address severe foreground-background class imbalance (9.85% anomaly pixels vs. 90.15% background), and benchmark three distinct deep learning architectures: (1) a baseline Convolutional U-Net trained from scratch, (2) a Transfer Learning U-Net with an ImageNet-pretrained ResNet-34 backbone, and (3) an unsupervised Convolutional Autoencoder evaluating reconstruction error residuals. Evaluated on unseen test frames, the Transfer Learning U-Net achieves a **Test IoU of 0.4833** and **Dice Score of 0.6449** with **95.47% pixel accuracy**, outperforming the baseline from-scratch network (IoU 0.1163, Dice 0.2083) by over $300\%$. Our findings demonstrate the critical role of pre-trained inductive bias and multi-objective loss engineering when learning from compact, high-variance safety datasets.

---

### 1. Introduction & Problem Statement
Detecting out-of-distribution (OOD) obstacles and anomalies is one of the most critical open challenges in autonomous vehicle perception. While modern deep semantic segmentation models perform exceptionally well on common urban classes (e.g., roads, sidewalks, pedestrians, cars in the Cityscapes dataset), they frequently fail catastrophically when encountering unusual obstacles (e.g., a herd of sheep, a lost tire, or a boulder). Failing to detect such objects can lead to severe accidents.

The objective of this project is to implement, evaluate, and critically analyze deep learning models capable of localizing anomalous road hazards at the pixel level.

---

### 2. Dataset Exploration & Preprocessing

#### 2.1 Dataset Characteristics & Taxonomy
The `RoadAnomaly` dataset comprises 60 real-world driving frames. We performed an exploratory data analysis (EDA) across all frames:
- **Resolution Uniformity:** All 60 frames share an identical spatial resolution of $1280 \times 720$ pixels (16:9 HD) in `.webp` format.
- **Annotation Format:** Ground truth masks are stored as pixel-level semantic maps (`labels_semantic.png`), where pixel values are strictly binary:
  - `0` = Normal background and road surfaces.
  - `2` = Anomalous hazards (remapped to `1` for binary model training).

To understand the semantic variety, we categorized the 60 frames into five distinct hazard classes:
1. **Animals (31 frames, 51.7%):** Cows, sheep, zebras, deer, bears, wild boars, and donkeys.
2. **Obstacles & Rocks (15 frames, 25.0%):** Fallen boulders, rockslides, and cardboard Czech hedgehogs.
3. **Traffic Cones (8 frames, 13.3%):** Construction cones, manhole barriers, and lane markers.
4. **Lost Tires (3 frames, 5.0%):** Abandoned vehicle tires in driving lanes.
5. **Unusual Vehicles (3 frames, 5.0%):** Non-standard agricultural carts and overloaded cargo bicycles.

#### 2.2 Class Imbalance Analysis
A quantitative audit of the pixel labels revealed severe class imbalance:
- **Total Pixels Analyzed:** $55,296,000$ pixels.
- **Anomaly Pixels:** $5,446,413$ pixels (**9.85%** of the dataset).
- **Background Pixels:** $49,849,587$ pixels (**90.15%** of the dataset).
- **Per-Image Variability:** The anomaly area varies drastically from **0.37%** (a small distant tire) to **42.33%** (a large sheep herd across the road).

#### 2.3 Stratified Splitting Strategy
Due to the small sample size and skewed categories (e.g., *Lost Tires* and *Unusual Vehicles* have only 3 instances each), random splitting would result in empty test classes. We designed a deterministic **Stratified Split**:
- **Training Set:** 40 frames ($66.7\%$)
- **Validation Set:** 10 frames ($16.7\%$)
- **Test Set:** 10 frames ($16.7\%$)
Every subset contains samples from all five hazard categories, ensuring balanced evaluation.

#### 2.4 Data Augmentation Pipeline
To mitigate overfitting on 40 training images, we standardized dimensions to $384 \times 512$ (exact multiples of 32 for encoder-decoder downsampling) and applied an augmentation pipeline using `Albumentations`:
- Random Horizontal Flip ($p=0.5$)
- Affine Transforms (rotation $\pm 15^\circ$, scaling $0.9-1.1$, translation $\pm 6\%$, $p=0.5$)
- Color Jitter (brightness, contrast, saturation $\pm 0.2$, $p=0.4$)
- Gaussian Noise injection ($p=0.25$)
- ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

### 3. Methodology & Model Architectures

We investigated three complementary deep learning paradigms:

```
               +----------------------------------------------------+
               |                Input Road Image                    |
               +-------------------------+--------------------------+
                                         |
         +-------------------------------+------------------------------+
         |                               |                              |
         v                               v                              v
+------------------+           +--------------------+         +-------------------+
| Baseline U-Net   |           | Transfer Learning  |         | Conv Autoencoder  |
| (From Scratch)   |           | ResNet-34 U-Net    |         | (Reconstruction)  |
+--------+---------+           +---------+----------+         +---------+---------+
         |                               |                              |
         v                               v                              v
[Predicted Mask 1]             [Predicted Mask 2]             [Pixel Residual Map]
```

#### 3.1 Model 1: Baseline Convolutional U-Net (From Scratch)
A standard U-Net architecture consisting of:
- **Encoder:** 4 stages of Double Convolution blocks (Conv $3 \times 3 \to$ BatchNorm $\to$ ReLU $\to$ Conv $3 \times 3 \to$ BatchNorm $\to$ ReLU) followed by $2 \times 2$ Max Pooling. Channels expand from $32 \to 64 \to 128 \to 256$.
- **Bottleneck:** 512 channels.
- **Decoder:** 4 transposed convolution upsampling stages with skip connections concatenating corresponding encoder feature maps.
- **Head:** $1 \times 1$ convolution producing single-channel anomaly logits.

#### 3.2 Model 2: Transfer Learning U-Net (Pre-trained ResNet-34)
Because 40 images are insufficient to learn robust edge, shape, and object representations from scratch, we employ a U-Net with an **ImageNet pre-trained ResNet-34 encoder**.
- Pre-trained weights provide low-level Gabor-like filters and mid-level texture detectors.
- The decoder is randomly initialized and trained end-to-end to assemble these general representations into anomaly segmentation maps.

#### 3.3 Model 3: Convolutional Autoencoder (Unsupervised Reconstruction)
As an unsupervised alternative, we implemented a 4-stage Convolutional Autoencoder:
- Trained strictly on image reconstruction with an L1 reconstruction loss: $\mathcal{L}_{\text{recon}} = ||x - \hat{x}||_1$.
- At test time, an **Anomaly Heatmap** is computed as the per-pixel reconstruction residual:
  $$\mathcal{A}(u, v) = \frac{1}{C} \sum_{c=1}^C |x_{c}(u, v) - \hat{x}_{c}(u, v)|$$
- Pixels with high reconstruction errors (top 10th percentile) correspond to unexpected obstacles that the autoencoder failed to reconstruct.

#### 3.4 Multi-Objective Loss Formulation: Combo Loss
Standard Binary Cross-Entropy (BCE) fails under 90:10 class imbalance because the model can achieve 90% accuracy by predicting all background. We implemented **Combo Loss**:
$$\mathcal{L}_{\text{Combo}} = \alpha \mathcal{L}_{\text{BCE}} + \beta \mathcal{L}_{\text{Dice}} + \gamma \mathcal{L}_{\text{Focal}}$$
- $\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$ directly optimizes spatial intersection.
- $\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ down-weights well-classified background pixels, focusing gradients on hard hazard boundaries.
- Hyperparameters: $\alpha = 0.4, \beta = 1.0, \gamma = 0.6$.

---

### 4. Experimental Setup & Results

#### 4.1 Training Configuration
- **Hardware:** NVIDIA GeForce RTX 3050 (6GB VRAM) running PyTorch 2.6.0 with CUDA 12.4.
- **Optimizer:** AdamW ($\text{weight\_decay} = 10^{-4}$) with Cosine Annealing learning rate scheduling ($\text{lr} = 3 \times 10^{-4} \to 10^{-6}$).
- **Batch Size:** 4.
- **Epochs:** 25 epochs for segmentation models; 20 epochs for autoencoder.

#### 4.2 Quantitative Test Set Evaluation (Unseen 10 Frames)

| Model Approach | Paradigm | Test IoU | Test Dice (F1) | Test Precision | Test Recall | Pixel Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** | Supervised (Scratch) | 0.1163 | 0.2083 | 0.1407 | 0.4680 | 78.48% |
| **Transfer Learning U-Net (ResNet-34)** | **Supervised (Pretrained)** | **0.4833** | **0.6449** | **0.6959** | **0.6446** | **95.47%** |
| **Convolutional Autoencoder** | Unsupervised (Residuals) | 0.1554 | 0.2472 | 0.2585 | 0.3255 | 88.13% |

---

### 5. Critical Analysis & Discussion

#### 5.1 The Impact of Transfer Learning
The empirical gap between the Baseline U-Net and Transfer Learning U-Net is striking:
- The baseline from scratch plateaued at **0.1163 IoU** and **0.2083 Dice**. With only 40 training images, the network was unable to learn generalizable visual primitives, leading to severe false positives and blurry predictions.
- The pre-trained ResNet-34 U-Net reached **0.4833 IoU** and **0.6449 Dice**, representing a **+315% increase in IoU** and a **+210% increase in Dice score**, while achieving **95.47% pixel accuracy**.
- **Conclusion:** On small-sample domain benchmarks, transfer learning is not merely an optimization; it is a fundamental requirement to provide visual priors and prevent overfitting.

#### 5.2 Supervised Segmentation vs. Reconstruction Residuals
- The Convolutional Autoencoder demonstrated an interesting capability: without ever seeing a single segmentation mask during training, its reconstruction error map achieved **0.1554 IoU** (outperforming the randomly initialized baseline U-Net).
- High-contrast anomalies (such as orange traffic cones and dark boulders on asphalt) produce prominent reconstruction residuals.
- However, reconstruction methods struggle with semantic boundaries: complex background textures (trees, roadside gravel, shadows) also produce high reconstruction errors, leading to lower precision (0.2585) compared to the supervised ResNet-34 (0.6959).

#### 5.3 Limitations & Failure Modes
1. **Camouflaged & Low-Contrast Obstacles:** Animals whose coats match road gravel or shadows (e.g., dark boars in shade or grey donkeys on dusty roads) experienced lower recall.
2. **Extreme Small-Object Scale:** Road debris or tires occupying less than $1\%$ of the image area are susceptible to being smoothed out by successive downsampling layers.
3. **Dataset Scale:** 60 frames remain a challenging benchmark; incorporating synthetic data or pre-training on broader datasets (like Cityscapes or Pascal VOC) would further elevate robustness.

---

### 6. Summary & Recommendations

1. **Adopt Pre-trained Encoders:** Pre-trained weights (ResNet-34, EfficientNet, MobileNet) are essential for autonomous driving edge-case perception when anomaly training data is scarce.
2. **Enforce Compound Losses:** Combining BCE with Dice and Focal losses prevents the network from collapsing to all-background predictions in the presence of $90\%$ background pixels.
3. **Hybrid Architecture Potential:** A promising future direction is combining the supervised Transfer U-Net with Autoencoder uncertainty maps, triggering emergency braking only when both semantic confidence and reconstruction error agree.

---

### References
- Lis, K., Nakka, K., Fua, P., and Salzmann, M. (2019). *Detecting the Unexpected via Image Resynthesis*. International Conference on Computer Vision (ICCV).
- Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV.
