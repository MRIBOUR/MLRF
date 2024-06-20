from . import feature_extraction
from . import models
from . import cifar_utils

__version__ = '1.0.0'

"""
mlrf - Machine Learning Research Framework

This package provides tools for feature extraction, model handling, and utilities for handling CIFAR-10 dataset.

Submodules:
- features: Classes for different feature extraction methods.
- models: Model definitions and utilities.
- cifar_utils: Utilities for finding and loading the CIFAR-10 dataset.

Example usage:
    from mlrf.feature_extraction import BoVWFE
"""

__all__ = ['feature_extraction', 'models', 'cifar_utils']