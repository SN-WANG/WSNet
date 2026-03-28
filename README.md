# WSNet

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**WSNet** is a compact, source-first library for engineering surrogate modeling and operator learning.  
It keeps the reusable core of the original project: surrogate models, neural models, optimization helpers, sampling methods, training utilities, and shared runtime tools.

## 📌 Overview

WSNet is designed to be cloned and used directly from the source tree.
The goal is not to be a feature-heavy framework. The goal is to keep the core modules clean, reusable, and easy to extend across multiple engineering ML projects.

The current repository focuses on four things:

- reusable surrogate models
- reusable neural and operator models
- lightweight sampling and optimization utilities
- lightweight training and utility modules

## ✨ Highlights

- Classical surrogate models: `PRS`, `RBF`, `KRG`, `SVR`
- Multi-fidelity surrogate models: `MFSMLS`, `MMFS`, `CCAMFS`
- Ensemble surrogate models: `TAHS`, `AESMSI`
- Neural models: `MLP`, `DeepONet`, `GeoFNO`, `HyperFlowNet`, `Transolver`
- Optimization utilities: `MIGA`, `CFSSDA`
- Sampling utilities: LHS, single-objective infill, multi-objective infill, multi-fidelity infill
- Training utilities for rollout-based and general learning workflows

## 🧱 Repository Layout

```text
WSNet/
├── __init__.py
├── models/
│   ├── classical/
│   ├── ensemble/
│   ├── multi_fidelity/
│   ├── neural/
│   └── optimization/
├── sampling/
├── training/
├── utils/
├── README.md
└── LICENSE
```

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/SN-WANG/WSNet.git
cd WSNet
```

### Install the dependencies you need

```bash
pip install numpy scipy
pip install torch
pip install matplotlib tqdm
```

WSNet is intended to be used directly from the cloned repository.
Run your scripts from the repository root, or add the repository root to `PYTHONPATH`.

### Minimal example

```python
import numpy as np

from models.classical.krg import KRG

x_train = np.random.rand(20, 2)
y_train = np.sum(x_train, axis=1, keepdims=True)

model = KRG()
model.fit(x_train, y_train)

y_pred, y_var = model.predict(x_train)
print(y_pred.shape, y_var.shape)
```

## 🧠 Module Guide

### `models/`

The core model collection.
This folder contains the reusable implementations of classical surrogates, multi-fidelity models, ensembles, neural models, and optimization routines.

### `sampling/`

Utilities for design of experiments and adaptive sampling.
This includes LHS generation and infill methods for single-objective, multi-objective, and multi-fidelity settings.

### `training/`

Lightweight trainer code for reusable learning workflows.
The focus is on compact training logic rather than large framework abstractions.

### `utils/`

Shared helpers for scaling, seeding, logging, and sweeps.

## 🎯 Design Philosophy

- Keep the repository small and reusable.
- Prefer readable implementations over heavy abstractions.
- Keep model code close to the math.
- Use consistent `fit` / `predict` style APIs where possible.
- Leave benchmark- and experiment-specific scripts to sibling repositories such as `SurrogateLab`.

## 🔗 Related Repository

If you want benchmark scripts, case studies, and research-facing experiment entry points, see **SurrogateLab**:

- [SurrogateLab](https://github.com/SN-WANG/SurrogateLab)

## 📚 Citation

If WSNet is useful in your work, please cite it as a software project.

```bibtex
@software{wsnet2026,
  author = {Shengning Wang},
  title = {WSNet},
  year = {2026},
  url = {https://github.com/SN-WANG/WSNet}
}
```

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## 📬 Contact

- Shengning Wang
- Email: `snwang2023@163.com`
- GitHub: [SN-WANG](https://github.com/SN-WANG)
