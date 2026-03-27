# wsnet/__init__.py
"""
WSNet: A Deep Learning Library for Engineering Surrogate Modeling.

WSNet is an integrated deep learning library specifically designed for high-fidelity surrogate modeling
in engineering applications. It provides a unified pipeline for fluid dynamics emulation, structural
analysis, and design optimization with comprehensive support for classical surrogate models, neural
networks, and modern neural operator algorithms.

The library follows a consistent "initialize, fit, predict" pattern across all models and training
frameworks, making it easy to use for both research and production applications.

Key Features:
- CFD-Ready Pipeline: Direct ingestion of ANSYS Fluent data with automatic coordinate and field mapping
- Physics-Informed Training: Support for physics constraints and loss functions
- Multi-Fidelity Support: Comprehensive multi-fidelity modeling capabilities
- Ensemble Methods: Advanced ensemble techniques for improved accuracy
- Neural Operator Algorithms: Modern neural operator implementations for complex physics
- Comprehensive Visualization Tools: Built-in CFD visualization and rendering

System Architecture:
```
WSNet/
├── models/        # Surrogate models (classical, neural, multi-fidelity, ensemble)
├── training/      # Training frameworks and utilities
├── data/          # Data loading and preprocessing
├── sampling/      # Design of Experiments and infill strategies
└── utils/         # Core utilities
```

For detailed documentation and examples, please refer to the README.md file.
"""

__version__ = "2.2.26"
__author__ = "Shengning Wang (王晟宁)"
__email__ = "snwang2023@163.com"
__description__ = "A Deep Learning Library for Engineering Surrogate Modeling"
__url__ = "https://github.com/SN-WANG/wsnet"
__license__ = "MIT"
