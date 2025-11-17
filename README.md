# JAX-PID-NN: Particle Identification in Challenging Momentum Regions

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.6.0+-green.svg)](https://github.com/google/flax)

**High-performance JAX/Flax neural network for particle identification in high-energy physics**

</div>

---

## Overview

**JAX-PID-NN** is a high-performance neural network framework for **particle identification (PID)** optimised for challenging detector regions (1-2 GeV/c momentum) where TPC and TOF detectors provide overlapping signatures.

Built for **ALICE O2Physics** with:
- JAX/Flax JIT compilation (~10× speedup)
- Automatic class weighting for imbalanced data
- Intelligent missing value handling
- Background cleaning & quality filtering

### Supported Particles

**Pion (69%) • Kaon (4.7%) • Proton (13.7%) • Electron (12.4%)**

---

## ✨ Features

- **Flax Neural Network** with batch normalization & dropout
- **Missing Value Handling** - Zero-fills TOF/TPC gaps with indicator features
- **Background Cleaning** - DCA cuts, momentum filters, PID consistency checks
- **Class Weighting** - Handles 14.6× imbalanced particle distribution
- **Early Stopping** - Prevents overfitting
- **GPU/TPU Support** - Seamless hardware acceleration
- **Complete Evaluation** - ROC curves, confusion matrices, precision/recall
- **JIT Compilation** - Lightning-fast training with automatic differentiation

---

## Repository Structure

```
jax-pid-nn/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── jax_pid_nn_1-2gev.ipynb
├── src/
│   ├── model.py
│   ├── data_loader.py
│   ├── background_cleaning.py
│   ├── training.py
│   └── evaluation.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── results/
    ├── models/
    ├── plots/
    └── metrics/
```

---

## Dataset Format

**21 features** from ALICE detector (TPC, TOF, Bayesian PID):
- Momentum: `pt`, `eta`, `phi`
- TPC signals: `tpc_signal`, `tpc_nsigma_*`
- TOF data: `tof_beta`, `tof_nsigma_*`
- PID scores: `bayes_prob_*`
- Track quality: `dca_xy`, `dca_z`, `has_tpc`, `has_tof`
- Ground truth: `mc_pdg`

**Size:** ~4.16M particles, ~500 MB compressed

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/forynski/jax-pid-nn/issues)
- **Discussions:** [GitHub Discussions](https://github.com/forynski/jax-pid-nn/discussions)
- **Email:** robert.forynski@cern.ch

---

## Citation

```bibtex
@software{jax_pid_nn_2025,
  title={JAX-PID-NN: Particle Identification in Challenging Momentum Regions},
  author={Robert Forynski},
  year={2025},
  url={https://github.com/forynski/jax-pid-nn}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Acknowledgments

- **JAX/Flax Teams** - High-performance ML framework
- **ALICE Collaboration** - Physics guidance
- **scikit-learn** - Machine learning utilities
