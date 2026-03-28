# WSNet

[![Role](https://img.shields.io/badge/Role-Core%20Library-2d6cdf)](https://github.com/SN-WANG/WSNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**WSNet** is the core repository in this project family. It collects the reusable models, sampling methods, optimization modules, training utilities, and shared tools used across multiple engineering machine learning projects.

## 📌 Overview

WSNet keeps the common building blocks only.
It is the place for code that should stay reusable across different tasks, instead of being tied to one benchmark suite or one application workflow.

The current scope includes:

- classical surrogate models
- ensemble surrogate models
- multi-fidelity surrogate models
- neural and operator models
- global optimization utilities
- DOE and infill sampling methods
- lightweight training and utility modules

## ✨ Highlights

- Classical surrogates: `PRS`, `RBF`, `KRG`, `SVR`
- Multi-fidelity models: `MFSMLS`, `MMFS`, `CCAMFS`
- Ensemble models: `TAHS`, `AESMSI`
- Neural models: `MLP`, `DeepONet`, `GeoFNO`, `HyperFlowNet`, `Transolver`
- Optimization helpers: `MIGA`, `CFSSDA`
- Sampling utilities for LHS, single-objective infill, multi-objective infill, and multi-fidelity infill
- Lightweight training utilities for general and rollout-based workflows

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

WSNet is meant to be used directly from the cloned source tree.
Install the packages required by the modules you plan to use. A common setup is:

```bash
pip install numpy scipy torch matplotlib tqdm pyvista
```

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

## 🧠 Design Scope

WSNet is the reusable base.
Benchmark scripts, case-specific experiments, and research-facing workflows live in sibling repositories built on top of it.

## 🔗 Related Repositories

- [SurrogateLab](https://github.com/SN-WANG/SurrogateLab): surrogate modeling benchmarks, sampling studies, and optimization demos
- [HyperFlowNet](https://github.com/SN-WANG/HyperFlowNet): irregular-mesh autoregressive CFD prediction
- [StructFieldNet](https://github.com/SN-WANG/StructFieldNet): design-conditioned structural field reconstruction

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
