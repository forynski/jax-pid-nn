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

**JAX-PID-NN** is a comprehensive neural network framework for **particle identification (PID)** in ALICE at the LHC, optimized for the challenging **0.7–3 GeV/c momentum range** where detector signatures overlap and missing data (especially TOF) significantly impacts traditional methods.

This repository includes **three complementary architectures**, from fast baseline to state-of-the-art:

1. **SimpleNN:** Fast, lightweight baseline
2. **DNN:** Deeper network with batch normalization
3. **FSE+Attention:** State-of-the-art with detector masking and attention mechanisms

Built for **ALICE O2Physics** with:
- JAX/Flax JIT compilation (~10× speedup)
- Focal Loss for class imbalance handling
- Intelligent missing data handling via detector masking
- Production-ready model persistence
- Comprehensive evaluation metrics

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
| **0.7-1.5 GeV/c Accuracy** | 86.6% |
| **1-3 GeV/c Accuracy** | 81.6% |
| **Inference Time** | ~0.2 ms/track |
| **Model Size** | ~1.2 MB |
| **Use Case** | Fast baseline, online monitoring |

### 2. DNN – Deeper with BatchNorm

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
| **0.7-1.5 GeV/c Accuracy** | 85.6% |
| **1-3 GeV/c Accuracy** | 80.3% |
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
| **0.7-1.5 GeV/c Accuracy** | **89.2%** |
| **1-3 GeV/c Accuracy** | **82.4%** |
| **Full Spectrum Macro-AUC** | **0.9280** |
| **Inference Time** | ~0.4 ms/track |
| **Model Size** | ~1.8 MB |
| **Use Case** | **Production – handles missing detectors elegantly** |
| **Advantage vs SimpleNN** | **+6.0%** (full), **+3.6%** (0.7-1.5), **+2.1%** (1-3) |

---

## Model Comparison

| Aspect | SimpleNN | DNN | FSE+Attention |
|--------|----------|-----|---------------|
| **Full Spectrum** | 85.8% | 86.8% | **92.8%** |
| **0.7-1.5 GeV/c** | 86.6% | 85.6% | **89.2%** |
| **1-3 GeV/c** | 81.6% | 80.3% | **82.4%** |
| **Speed** | Fastest | Medium | Slower |
| **Memory** | ~1.2 MB | ~2.1 MB | ~1.8 MB |
| **Handles Missing Data** | Assumes complete | Assumes complete | Explicit masking |
| **Best For** | Real-time inference | Balanced approach | Production accuracy |

---

## Key Features

### Advanced Training Techniques

- **Focal Loss:** Down-weights easy examples → +2–3% on minority classes
- **Class Weighting:** Balanced handling of imbalanced data (π:K:p:e ≈ 15:1:3:3)
- **Early Stopping:** Prevents overfitting with patience=10
- **Batch Normalization (DNN):** Stabilizes deep networks
- **Feature Set Embedding (FSE+Attention):** Adaptive detector grouping

### Missing Data Handling

**SimpleNN & DNN:** Fill missing values with zeros/medians (standard approach)

**FSE+Attention:** Explicit detector masking with attention
- Tracks which detectors are available per particle
- Learns adaptive importance of each detector group
- Handles extreme TOF scarcity (8.5%) gracefully
- **Result:** 3–6% improvement in challenging momentum ranges

### Production Ready

- **JIT Compilation:** JAX automatic optimization (~10× speedup)
- **GPU/TPU Support:** Seamless hardware acceleration
- **Model Persistence:** Two-tier save/load from Kaggle
- **Comprehensive Metrics:** ROC curves, confusion matrices, per-class F1 scores
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

*Estimated - actual values depend on your dataset

**Key Insight:** TOF only 8.5% in critical 0.7-1.5 GeV/c range → FSE+Attention learns to upweight TPC when TOF missing

---

## Per-Class Performance (FSE+Attention, Full Spectrum)

| Particle | AUC | F1 Score | Notes |
|---|---|---|---|
| **Pion** | 0.9050 | 0.92 | Abundant, excellent performance |
| **Kaon** | 0.8938 | 0.78 | Most challenging (π/K confusion) |
| **Proton** | 0.9793 | 0.95 | Easiest to identify |
| **Electron** | 0.9340 | 0.91 | Unique detector signature |

---

## Features

- **Three Neural Network Architectures** - Choose based on accuracy/speed tradeoff
- **Focal Loss Training** - Better handling of class imbalance
- **Detector Masking** - FSE+Attention handles missing data explicitly
- **Batch Normalization (DNN)** - Stabilizes training
- **Early Stopping** - Prevents overfitting with patience=10
- **GPU/TPU Support** - Seamless hardware acceleration
- **JIT Compilation** - ~10× speedup via JAX optimization
- **Complete Evaluation** - ROC curves, confusion matrices, F1 scores, AUC
- **Model Persistence** - Two-tier save/load system
- **Momentum-Specific Training** - Separate models for 3 momentum ranges

---

## Dataset Format

**21 features** from ALICE detector (TPC, TOF, Bayesian PID):
- **Momentum:** `pt`, `eta`, `phi`
- **TPC signals:** `tpc_signal`, `tpc_nsigma_*`
- **TOF data:** `tof_beta`, `tof_nsigma_*`
- **PID scores:** `bayes_prob_*`
- **Track quality:** `dca_xy`, `dca_z`, `has_tpc`, `has_tof`
- **Ground truth:** `mc_pdg`

### Statistics

- **Size:** ~4.16M particles
- **Momentum Range:** 0.1 – 10 GeV/c
- **Class Distribution:** π (69%), K (5%), p (14%), e (12%)
- **Imbalance Ratio:** ~14.6:1 (majority:minority)
- **Source:** ALICE Pb-Pb Run 3 Monte Carlo

---

## Evaluation Metrics

Computed for all three models:
- **Accuracy:** Per-class and macro-average
- **ROC Curves:** Per-class and macro-averaged
- **AUC Scores:** Macro and micro-average
- **Confusion Matrix:** Normalized (true rates)
- **F1 Scores:** Per-particle species
- **Training Curves:** Loss and validation accuracy

---

## References

### Academic Papers

1. **Focal Loss:** [Lin et al., 2017](https://arxiv.org/abs/1708.02002) - "Focal Loss for Dense Object Detection"
2. **ALICE PID ML:** [arXiv:2309.07768](https://arxiv.org/abs/2309.07768) - "Particle identification with machine learning in ALICE Run 3"
3. **Missing Data:** [arXiv:2403.17436](https://arxiv.org/abs/2403.17436) - "Missing data handling in ML for particle identification"
4. **Attention:** [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) - "Attention is All You Need"

### ALICE Resources

- [ALICE O2Physics](https://github.com/AliceO2Group/O2Physics)
- [ALICE PID ML Tools](https://github.com/AliceO2Group/O2Physics/tree/master/Tools/PIDML)
- [ALICE Analysis Tutorial](https://alice-analysis-tutorial.readthedocs.io/)

---

## Citation

```bibtex
@software{jax_pid_nn_2025,
  title={JAX-PID-NN: Particle Identification in Challenging Momentum Regions},
  author={Forynski, Robert},
  year={2025},
  url={https://github.com/forynski/jax-pid-nn},
  note={Three architectures: SimpleNN, DNN, and FSE+Attention}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file

```
MIT License
Copyright (c) 2025 Robert Forynski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Contact & Support

- **Email:** [robert.forynski@cern.ch](mailto:robert.forynski@cern.ch)
- **Issues:** [GitHub Issues](https://github.com/forynski/jax-pid-nn/issues)
- **Discussions:** [GitHub Discussions](https://github.com/forynski/jax-pid-nn/discussions)
- **Institution:** CERN, ALICE Collaboration

---

## Acknowledgments

- **JAX/Flax Teams** for high-performance ML infrastructure
- **ALICE Collaboration** for physics expertise and data
- **scikit-learn Contributors** for machine learning utilities
