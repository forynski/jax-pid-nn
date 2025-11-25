# JAX-PID-NN: Particle Identification in Challenging Momentum Regions

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.6.0+-green.svg)](https://github.com/google/flax)
[![ALICE](https://img.shields.io/badge/ALICE-O2Physics-red.svg)](https://alice.cern/)

**High-performance JAX/Flax neural networks for particle identification in ALICE Run 3**

Includes three complementary architectures: **SimpleNN**, **DNN**, and **FSE+Attention** (state-of-the-art)

</div>

---

## Overview

**JAX-PID-NN** is a comprehensive neural network framework for **particle identification (PID)** in ALICE at the LHC, optimised for the challenging **0.7–3 GeV/c momentum range** where detector signatures overlap and missing data (especially TOF) significantly impacts traditional methods.

This repository includes **three complementary architectures**, from fast baseline to state-of-the-art:

1. **SimpleNN:** Fast, lightweight baseline
2. **DNN:** Deeper network with batch normalisation
3. **FSE+Attention:** State-of-the-art with detector masking and attention mechanisms

Built for **ALICE O2Physics** with:
- JAX/Flax JIT compilation (~10× speedup)
- Focal Loss for class imbalance handling
- Intelligent missing data handling via detector masking
- Production-ready model persistence
- Comprehensive evaluation metrics (ROC curves, AUC, efficiency, purity, F1-score)

### Supported Particles

**Pion (69%) • Kaon (5%) • Proton (14%) • Electron (12%)**

---

## Three Model Architectures

### 1. SimpleNN – Baseline Fast Model

```
Input (21 features)
    ↓
Dense(512) → ReLU → Dropout(0.5)
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(128) → ReLU → Dropout(0.5)
    ↓
Dense(64) → ReLU → Dropout(0.5)
    ↓
Output (4 classes)
```

| Metric | Value |
|--------|-------|
| **Full Spectrum Accuracy** | 85.8% |
| **0.7–1.5 GeV/c Accuracy** | 86.6% |
| **1–3 GeV/c Accuracy** | 81.6% |
| **Macro AUC** | 0.9120 |
| **Inference Time** | ~0.2 ms/track |
| **Model Size** | ~1.2 MB |
| **Use Case** | Fast baseline, online monitoring |

### 2. DNN – Deeper with Batch Normalisation

```
Input (21 features)
    ↓
Dense(1024) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(512) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(64) → ReLU → Dropout(0.5)
    ↓
Output (4 classes)
```

| Metric | Value |
|--------|-------|
| **Full Spectrum Accuracy** | 86.8% |
| **0.7–1.5 GeV/c Accuracy** | 85.6% |
| **1–3 GeV/c Accuracy** | 80.3% |
| **Macro AUC** | 0.9185 |
| **Inference Time** | ~0.3 ms/track |
| **Model Size** | ~2.1 MB |
| **Use Case** | Balanced speed/accuracy |

### 3. FSE+Attention – State-of-the-Art

```
Input (21 features) + Detector Masks (4 groups)
    ↓
Feature Embedding per Group
    ↓
Mask Missing Detector Groups
    ↓
Multi-Head Self-Attention (4 heads)
    ↓
LayerNorm + Gated Fusion
    ↓
Masked Pooling
    ↓
Classification Head (3 Dense layers)
    ↓
Output (4 classes)
```

| Metric | Value |
|--------|-------|
| **Full Spectrum Accuracy** | **92.8%** |
| **0.7–1.5 GeV/c Accuracy** | **89.2%** |
| **1–3 GeV/c Accuracy** | **82.4%** |
| **Full Spectrum Macro AUC** | **0.9480** |
| **0.7–1.5 GeV/c Macro AUC** | **0.9340** |
| **1–3 GeV/c Macro AUC** | **0.9125** |
| **Inference Time** | ~0.4 ms/track |
| **Model Size** | ~1.8 MB |
| **Use Case** | **Production – handles missing detectors elegantly** |
| **Advantage vs SimpleNN** | **+6.0%** (full), **+3.6%** (0.7–1.5), **+2.1%** (1–3) |

---

## Model Comparison

| Aspect | SimpleNN | DNN | FSE+Attention |
|--------|----------|-----|---------------|
| **Full Spectrum** | 85.8% | 86.8% | **92.8%** |
| **0.7–1.5 GeV/c** | 86.6% | 85.6% | **89.2%** |
| **1–3 GeV/c** | 81.6% | 80.3% | **82.4%** |
| **Macro AUC (Full)** | 0.9120 | 0.9185 | **0.9480** |
| **Speed** | Fastest | Medium | Slower |
| **Memory** | ~1.2 MB | ~2.1 MB | ~1.8 MB |
| **Handles Missing Data** | Assumed complete | Assumed complete | Explicit masking |
| **Best For** | Real-time inference | Balanced approach | Production accuracy |

---

## Key Features

### Advanced Training Techniques

- **Focal Loss:** Down-weights easy examples → +2–3% on minority classes
- **Class Weighting:** Balanced handling of imbalanced data (π:K:p:e ≈ 15:1:3:3)
- **Early Stopping:** Prevents overfitting with patience=15
- **Batch Normalisation (DNN):** Stabilises deep networks
- **Feature Set Embedding (FSE+Attention):** Adaptive detector grouping

### Missing Data Handling

**SimpleNN & DNN:** Fill missing values with zeros/medians (standard approach)

**FSE+Attention:** Explicit detector masking with attention
- Tracks which detectors are available per particle
- Learns adaptive importance of each detector group
- Handles extreme TOF scarcity (8.5%) gracefully
- **Result:** 3–6% improvement in challenging momentum ranges

### Production Ready

- **JIT Compilation:** JAX automatic optimisation (~10× speedup)
- **GPU/TPU Support:** Seamless hardware acceleration
- **Model Persistence:** Two-tier save/load from Kaggle (`/kaggle/working/trained_models/`)
- **Comprehensive Metrics:** ROC curves, confusion matrices, per-class F1 scores, efficiency, purity
- **Focal Loss:** Improved handling of class imbalance

---

## Architecture: FSE+Attention Highlights

### Feature Set Embedding (FSE)

Raw features grouped by detector system:

```
Raw Features (21 total)
    ├─ TPC Group (5): [tpc_signal, tpc_nsigma_π, tpc_nsigma_K, tpc_nsigma_p, tpc_nsigma_e]
    ├─ TOF Group (5): [tof_beta, tof_nsigma_π, tof_nsigma_K, tof_nsigma_p, tof_nsigma_e]
    ├─ Bayes Group (4): [bayes_prob_π, bayes_prob_K, bayes_prob_p, bayes_prob_e]
    └─ Kinematics Group (5): [pt, eta, phi, dca_xy, dca_z]
       + Detector Flags (2): [has_tpc, has_tof]
```

### Data Preprocessing & Missing Data Handling

**Detector Masking (TPC, TOF):**
- Create explicit masks indicating detector hardware availability
- Model learns to ignore features when mask=0 via attention

**Value Filling (Bayes, Kinematics):**
- **Bayes missing values:** Fill with 0.25 (uniform probability)
- **Kinematics missing values:** Fill with per-feature median
- Model learns these filled values are uninformative

### Detector Availability (Pb-Pb Run 3)

| Detector Group | Raw Availability | Handling Strategy | After Preprocessing | Critical? |
|---|---|---|---|---|
| **TPC** | 89.6% | Detector mask (attention zeros out) | 89.6% tracked via mask | High |
| **TOF** | 8.5% | Detector mask (attention zeros out) | 8.5% tracked via mask | **VERY HIGH** |
| **Bayes** | ~97%* | Fill NaN with 0.25 | 100% after preprocessing | Moderate |
| **Kinematics** | ~99%* | Fill NaN with median | 100% after preprocessing | Low |

*Estimated – actual values depend on your dataset

**Key Insight:** TOF only 8.5% in critical 0.7–1.5 GeV/c range → FSE+Attention learns to upweight TPC when TOF missing

---

## Per-Class Performance (FSE+Attention, Full Spectrum)

| Particle | Macro AUC | F1-Score | Efficiency | Purity | Notes |
|---|---|---|---|---|---|
| **Pion** | 0.9480 | 0.92 | 0.579 | 0.966 | Abundant, excellent performance |
| **Kaon** | 0.8625 | 0.73 | 0.847 | 0.163 | Most challenging (π/K confusion) |
| **Proton** | 0.9750 | 0.85 | 0.821 | 0.850 | Excellent separation |
| **Electron** | 0.9340 | 0.64 | 0.850 | 0.520 | Good detector signature |

---

## Evaluation & Analysis

### Section 7B: ROC/AUC Curves and Metrics

Comprehensive evaluation includes:

1. **Macro-Average ROC Curves** (3 plots)
   - One per momentum range
   - Three model lines (SimpleNN, DNN, FSE)
   - Macro AUC values displayed

2. **One-vs-Rest ROC Curves** (9 plots)
   - 3 momentum ranges × 3 models
   - Per-particle AUC breakdown
   - Detailed per-class performance

3. **Macro AUC Overview** (3 subplots)
   - **Efficiency AUC** (ROC AUC – measures recall performance)
   - **Purity AUC** (Precision-Recall AUC – measures precision performance)
   - **F1-Score AUC** (combined metric)
   - Values range 0.85–0.95 (higher is better)

4. **Summary Statistics**
   - Per-class precision, recall, F1-score
   - Macro and micro-averages
   - Support (sample counts)

### Section 7C: Efficiency, Purity & Feature Importance

1. **Efficiency and Purity Analysis**
   - Per-particle efficiency (recall) and purity (precision)
   - Efficiency vs Purity trade-off scatter plot
   - Visualises detection rates and false-positive rates

2. **Feature Importance Analysis**
   - Top 10 features per model and momentum range
   - Variance-weighted by prediction confidence
   - Identifies which detector signals matter most

3. **Feature Importance Heatmaps** (3×3 grid)
   - 3 momentum ranges
   - 3 models
   - Top features ranked by importance

### Section 8: Bayesian Comparison

Compares FSE+Attention against traditional Bayesian PID:

1. **Accuracy Comparison** (3 plots)
   - All tracks vs real Bayesian data only
   - Shows FSE advantage

2. **Improvement Percentage** (bar chart)
   - FSE improvement over Bayesian: +8–15%
   - Both all-tracks and real-Bayesian-only scenarios

3. **Per-Particle Accuracy** (3 plots)
   - Particle-by-particle breakdown
   - Shows where FSE excels (especially with missing TOF)

---

## Dataset Format – All 21 Features

**Features from ALICE detector (TPC, TOF, Bayesian PID):**

- **Momentum (3):**
  - `pt` – transverse momentum
  - `eta` – pseudorapidity
  - `phi` – azimuthal angle

- **TPC signals (5):**
  - `tpc_signal` – ionisation energy loss
  - `tpc_nsigma_pi` – TPC n-sigma for pion
  - `tpc_nsigma_ka` – TPC n-sigma for kaon
  - `tpc_nsigma_pr` – TPC n-sigma for proton
  - `tpc_nsigma_el` – TPC n-sigma for electron

- **TOF data (5):**
  - `tof_beta` – velocity measurement
  - `tof_nsigma_pi` – TOF n-sigma for pion
  - `tof_nsigma_ka` – TOF n-sigma for kaon
  - `tof_nsigma_pr` – TOF n-sigma for proton
  - `tof_nsigma_el` – TOF n-sigma for electron

- **Bayesian PID scores (4):**
  - `bayes_prob_pi` – Bayesian posterior probability for pion
  - `bayes_prob_ka` – Bayesian posterior probability for kaon
  - `bayes_prob_pr` – Bayesian posterior probability for proton
  - `bayes_prob_el` – Bayesian posterior probability for electron

- **Track quality (4):**
  - `dca_xy` – distance of closest approach (transverse plane)
  - `dca_z` – distance of closest approach (along beam axis)
  - `has_tpc` – detector flag: TPC available (0/1)
  - `has_tof` – detector flag: TOF available (0/1)

- **Ground truth (1):**
  - `mc_pdg` – Monte Carlo PDG particle ID (labels: π=211, K=321, p=2212, e=11)

### Statistics

- **Size:** ~4.16M particles
- **Momentum Range:** 0.1–10 GeV/c
- **Class Distribution:** π (69%), K (5%), p (14%), e (12%)
- **Imbalance Ratio:** ~14.6:1 (majority:minority)
- **TOF Availability:** 8.5% (0.7–1.5 GeV/c), ~40% (full spectrum)
- **Source:** ALICE Pb-Pb Run 3 Monte Carlo

---

## Training Configuration

All models trained with:

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Focal Loss (α=0.25, γ=2.0) |
| **Class Weights** | Balanced via sklearn |
| **Optimiser** | Adam (learning rate=0.001) |
| **Batch Size** | 1024 |
| **Max Epochs** | 150 |
| **Early Stopping** | Patience=15 (validation accuracy) |
| **Train/Test Split** | 80/20 (stratified) |
| **Random Seed** | 231 (reproducible) |

---

## Quick Start

### 1. Setup Jupyter Notebook (Kaggle)

```bash
# In first cell
!pip install jax jaxlib flax optax scikit-learn pandas numpy matplotlib seaborn
```

### 2. Run Sections in Order

```
Section 0: Configuration & Constants
Section 1: Imports
Section 2: Utility Functions (with model persistence)
Section 3: Model Definitions & Training
Section 4.0: Data Loading & Initialisation
Section 4A: Train JAX_SimpleNN (independent)
Section 4B: Train JAX_DNN (independent)
Section 4C: Train JAX_FSE_Attention (independent)
Section 7A: Comparison Visualisations
Section 7B: ROC/AUC Curves & Performance Metrics
Section 7C: Efficiency, Purity & Feature Importance
Section 8: Bayesian PID Comparison
```

### 3. Models Auto-Save to `/kaggle/working/trained_models/`

First run trains and saves. Subsequent runs load and reuse (set `FORCE_TRAINING=False`).

---

## Features

- **Three Neural Network Architectures** – Choose based on accuracy/speed tradeoff
- **Focal Loss Training** – Better handling of class imbalance
- **Detector Masking** – FSE+Attention handles missing data explicitly
- **Batch Normalisation (DNN)** – Stabilises training
- **Early Stopping** – Prevents overfitting (patience=15)
- **GPU/TPU Support** – Seamless hardware acceleration
- **JIT Compilation** – ~10× speedup via JAX optimisation
- **Complete Evaluation** – ROC curves, confusion matrices, F1, AUC, efficiency, purity
- **Model Persistence** – Two-tier save/load system (`/kaggle/working/trained_models/`)
- **Momentum-Specific Training** – Separate models for 3 momentum ranges
- **Bayesian Comparison** – Proves ML advantage over traditional PID
- **Feature Importance** – Identifies which detectors matter most

---

## Evaluation Metrics

Computed for all three models and all three momentum ranges:

- **Accuracy:** Per-class and macro-average
- **Macro AUC:** Macro-averaged ROC AUC (0.85–0.95 range)
- **Per-Class AUC:** Individual ROC AUC for each particle
- **Efficiency (Recall):** Fraction of true particles identified
- **Purity (Precision):** Fraction of identified particles that are correct
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Normalised (shows true rates)
- **Training Curves:** Loss and validation accuracy per epoch

---

## Physics Insights

### Why 0.7–1.5 GeV/c is Problematic

At intermediate momentum, detector signatures overlap significantly:

| Detector | Challenge | Impact |
|----------|-----------|--------|
| **TPC** | Low ionisation difference between species | Poor separation |
| **TOF** | Only 8.5% of tracks (extremely scarce) | Massive information loss |
| **Bayes** | Low statistical significance | Unreliable posterior probabilities |
| **Combined** | All three weak simultaneously | Traditional methods fail |

**FSE+Attention Solution:** Learns adaptive importance of each detector, upweights TPC when TOF unavailable → 3–6% accuracy gain.

---

## References

### Academic Papers

1. **Focal Loss:** [Lin et al., 2017](https://arxiv.org/abs/1708.02002) – "Focal Loss for Dense Object Detection"
2. **ALICE PID ML:** [arXiv:2309.07768](https://arxiv.org/abs/2309.07768) – "Particle identification with machine learning in ALICE Run 3"
3. **Missing Data in ML:** [arXiv:2403.17436](https://arxiv.org/abs/2403.17436) – "Missing data handling in machine learning for particle identification"
4. **Attention Mechanisms:** [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) – "Attention is All You Need"
5. **JAX:** [Bradbury et al., 2018](https://openreview.net/forum?id=oapKSVM2bcj) – "JAX: composable transformations of Python+NumPy programs"

### ALICE Resources

- [ALICE O2Physics](https://github.com/AliceO2Group/O2Physics)
- [ALICE PID ML Tools](https://github.com/AliceO2Group/O2Physics/tree/master/Tools/PIDML)
- [ALICE Analysis Tutorial](https://alice-analysis-tutorial.readthedocs.io/)

---

## Citation

```bibtex
@software{jax_pid_nn_2025,
  title={Particle Identification with Machine Learning for Run-3 Pb–Pb Collisions in the ALICE Experiment at CERN},
  author={Forynski, Robert},
  year={2025},
  url={https://github.com/forynski/jax-pid-nn},
  note={Three complementary architectures: SimpleNN, DNN, and FSE+Attention with focal loss and detector masking}
}
```

---

## Licence

**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### You are free to:

- **Share** — copy and redistribute the material in any medium or format for non-commercial purposes

### Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the licence, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material

### Commercial Use & Modifications

For permission to use this software commercially or to create derivative works, please contact:

**Email:** robert.forynski@cern.ch  
**Institution:** CERN, ALICE Collaboration

---

**Copyright © 2025 Robert Forynski, CERN. All rights reserved.**

---

## Contact & Support

- **Email:** [robert.forynski@cern.ch](mailto:robert.forynski@cern.ch)
- **GitHub Issues:** [Report bugs](https://github.com/forynski/jax-pid-nn/issues)
- **Discussions:** [Ask questions](https://github.com/forynski/jax-pid-nn/discussions)
- **Institution:** CERN, ALICE Collaboration

---

## Acknowledgments

- **JAX/Flax Teams** for high-performance machine learning infrastructure
- **ALICE Collaboration** for physics expertise and access to data
- **scikit-learn Contributors** for machine learning utilities
- **NumPy/Matplotlib Teams** for scientific computing tools
