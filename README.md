# Eye Disease Classification with Classic CNNs

> **A Comparative Study of Deep Learning Architectures for Retinal Fundus Image Analysis**
>
> 🔬 Benchmarking LeNet, AlexNet, VGG16, GoogLeNet, and ResNet18 on the iChallenge-PM Dataset

---

## 🏥 Background

Eye diseases are among the leading causes of visual impairment worldwide. According to the **World Health Organization**, approximately **2.5 billion people** suffer from various ocular conditions globally. This project focuses on automated classification of retinal fundus images into three critical categories:

| Class | Description | Clinical Significance |
|-------|-------------|----------------------|
| **PM** | Pathologic Myopia | Severe structural changes requiring intervention |
| **H** | High Myopia | Elevated risk, needs monitoring |
| **N** | Normal | Healthy retinal condition |

Traditional diagnosis relies on ophthalmologists manually interpreting fundus images—a process that is:
- ⏱️ **Time-consuming:** Cannot scale for mass screening
- 🎯 **Subjective:** Prone to inter-observer variability
- 🌍 **Inaccessible:** Quality care unevenly distributed globally

**Our Goal:** Systematically evaluate classic CNN architectures to identify optimal models for small-scale medical imaging datasets, providing actionable guidance for clinical AI deployment.

---

## 🚀 The Challenge

### 1. Small Dataset, Big Models

The **iChallenge-PM** dataset contains only **400 training images**—a stark contrast to ImageNet's 1.2M samples. This creates a fundamental tension:

| Model | Parameters | Samples/Parameter | Risk Level |
|-------|------------|-------------------|------------|
| LeNet | ~0.3M | 1,067 | ⚠️ Underfitting |
| ResNet18 | ~11M | 0.03 | ✅ Balanced |
| VGG16 | ~138M | 0.002 | 🔴 Severe Overfitting |

### 2. Class Imbalance & Feature Overlap

- **PM vs H:** Both conditions involve axial elongation, creating overlapping feature distributions
- **Subtle Differences:** Early-stage pathological changes may be visually similar to high myopia
- **Clinical Stakes:** False negatives (missed PM) have serious consequences

### 3. No Pre-training Baseline

To ensure fair comparison, all models are trained **from scratch** (random initialization), exposing their raw learning capacity on limited medical data.

---

## 🛠️ Solution Framework

### 1. Data Pipeline

```
Raw Images (Variable Size)
    │
    ▼
┌─────────────────────────────────┐
│  Resize to 224×224              │
│  ToTensor [0,255] → [0,1]       │
│  Normalize (ImageNet μ, σ)      │
└─────────────────────────────────┘
    │
    ▼
Train/Val Split (80/20)
```

**Normalization Formula:**
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 2. Model Architectures

#### LeNet (1998) — *The Pioneer*
- **Structure:** 2 Conv → 2 Pool → 3 FC
- **Adapted for 224×224:** Feature map progression: 224→220→110→106→53
- **Strength:** Lightweight baseline (~0.3M params)
- **Weakness:** Limited receptive field for complex patterns

#### AlexNet (2012) — *The Breakthrough*
- **Innovations:** ReLU activation, Dropout regularization, LRN
- **Key Insight:** First to demonstrate deep learning superiority over handcrafted features
- **Dropout Rate:** $p = 0.5$ in FC layers

#### VGG16 (2014) — *Depth Through Simplicity*
- **Philosophy:** Stack 3×3 convolutions instead of large kernels
- **Math:** Two 3×3 convs = One 5×5 receptive field, but $18C^2 < 25C^2$ parameters
- **Reality Check:** 138M parameters demand massive data or pre-training

#### GoogLeNet (2014) — *Efficiency Through Width*
- **Inception Module:** Parallel 1×1, 3×3, 5×5 convolutions + pooling
- **Innovation:** 1×1 convolutions for dimensionality reduction
- **Trade-off:** Complex architecture, harder to train on small data

#### ResNet18 (2015) — *Depth Without Degradation*
- **Residual Connection:**
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
- **Gradient Flow:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$$
- **Why It Works:** Skip connections ensure gradients reach early layers

### 3. Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | Adam | Adaptive learning rates |
| Learning Rate | 0.001 | Standard initialization |
| Batch Size | 16 | Memory-gradient balance |
| Epochs | 10 | Prevent overfitting |
| Loss | CrossEntropyLoss | Multi-class classification |

**Cross-Entropy Loss:**
$$\mathcal{L} = -\sum_{c=1}^{3} y_c \log(\hat{y}_c)$$

### 4. Evaluation Metrics

We employ comprehensive metrics to capture different aspects of model performance:

- **Accuracy:** Overall correctness
- **Precision (Weighted):** $\sum_c \frac{n_c}{N} \cdot \frac{TP_c}{TP_c + FP_c}$
- **Recall (Weighted):** $\sum_c \frac{n_c}{N} \cdot \frac{TP_c}{TP_c + FN_c}$
- **F1 Score:** Harmonic mean balancing precision and recall

---

## 📊 Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| LeNet | 0.8875 | 0.9013 | 0.8875 | 0.8928 |
| AlexNet | **0.9375** | **0.9406** | **0.9375** | 0.9217 |
| VGG16 | 0.5750 | 0.3307 | 0.5750 | 0.4198 |
| GoogLeNet | 0.7750 | 0.8177 | 0.7750 | 0.7717 |
| ResNet18 | **0.9375** | 0.9307 | **0.9375** | **0.9318** |

### Key Findings

1. **🏆 ResNet18 & AlexNet** achieve identical accuracy (93.75%), but ResNet18 edges ahead on F1 score (0.9318 vs 0.9217)

2. **🔴 VGG16 Catastrophically Fails** — With only 0.002 samples per parameter, the model memorizes noise rather than learning features

3. **📉 Depth ≠ Performance** — GoogLeNet (22 layers) underperforms AlexNet (8 layers), proving architecture design matters more than raw depth

4. **✅ Residual Connections are Critical** — Enable stable training of deeper networks on limited data

### Training Dynamics

```
Loss Convergence (10 Epochs)
─────────────────────────────────────
ResNet18  : ████████████████░░░░ 0.30 ✓ Smooth descent
AlexNet   : ███████████████░░░░░ 0.40 ✓ Stable with Dropout
LeNet     : ██████████████░░░░░░ 0.60 ~ Limited capacity
GoogLeNet : █████████████░░░░░░░ 0.80 ~ Initial fluctuation
VGG16     : █████████░░░░░░░░░░░ 1.20 ✗ Stagnant (overfitting)
```

---

## 🧰 Tech Stack

### Deep Learning Framework
- **PyTorch** — Model definition, training loops, GPU acceleration
- **torchvision** — Pre-built architectures, transforms

### Scientific Computing
- **NumPy** — Numerical operations
- **Pandas** — Data manipulation and analysis

### Visualization
- **Matplotlib** — Training curves, bar charts
- **Seaborn** — Confusion matrix heatmaps

### Evaluation
- **scikit-learn** — Precision, recall, F1, confusion matrix

---

## 📁 Repository Structure

```
eye-disease-classification/
│
├── 📁 data/
│   └── PALM-Training400/          # iChallenge-PM dataset
│       ├── P0001.jpg ... P0133.jpg   # Pathologic Myopia
│       ├── H0001.jpg ... H0133.jpg   # High Myopia
│       └── N0001.jpg ... N0133.jpg   # Normal
│
├── 📁 notebooks/
│   └── eye_disease_classification.ipynb  # Complete training pipeline
│
├── 📁 paper/
│   ├── eye_disease_classification.tex    # LaTeX source
│   └── eye_disease_classification.pdf    # Compiled paper
│
├── 📁 figures/
│   ├── architecture/              # Model diagrams
│   ├── training_curves.png        # Loss over epochs
│   ├── confusion_matrices.png     # Per-model confusion matrices
│   └── metrics_comparison.png     # Bar chart comparison
│
└── README.md
```

---

## 🔮 Future Directions

### Data-Level Improvements
- **Data Augmentation:** Rotation, flipping, color jittering to expand effective dataset 3-5×
- **Transfer Learning:** Initialize with ImageNet weights to leverage pre-learned features

### Model-Level Enhancements
- **Attention Mechanisms:** Add CBAM to focus on lesion-relevant regions (macula, optic disc)
- **Ensemble Methods:** Combine ResNet18 + AlexNet predictions for potential >95% accuracy
- **Lightweight Architectures:** MobileNet/ShuffleNet for edge deployment

### Clinical Validation
- **Multi-Center Testing:** Validate on datasets from different hospitals and imaging equipment
- **Interpretability:** Generate Grad-CAM heatmaps to explain model decisions to clinicians

---

## 📚 References

1. LeCun, Y., et al. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.

2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks.* NeurIPS.

3. Simonyan, K., & Zisserman, A. (2015). *Very deep convolutional networks for large-scale image recognition.* ICLR.

4. Szegedy, C., et al. (2015). *Going deeper with convolutions.* CVPR.

5. He, K., et al. (2016). *Deep residual learning for image recognition.* CVPR.

---

## 📜 License

This project is released under the MIT License.

---

<p align="center">
  <i>Built with ❤️ for advancing AI in ophthalmology</i>
</p>
