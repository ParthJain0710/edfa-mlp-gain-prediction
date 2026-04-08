# EDFA Gain Spectrum Decomposition – MLP Models

Machine learning project developed as part of a research module at **TU Dortmund University** (Master's in Automation and Robotics).

## Overview

This project trains Multi-Layer Perceptron (MLP) neural networks to decompose EDFA (Erbium-Doped Fiber Amplifier) gain spectra into spectral basis functions. Each model predicts the parameters of a specific spectral shape used to reconstruct the gain profile of an optical amplifier.

## Model Variants

Nine model configurations are implemented across three spectral basis types and three peak counts:

| Basis Function | 8 Peaks | 10 Peaks | 12 Peaks |
|---|---|---|---|
| Gaussian | ✓ | ✓ | ✓ |
| Lorentzian | ✓ | ✓ | ✓ |
| Pseudo-Voigt | ✓ | ✓ | ✓ |

## Pipeline

Each model folder contains the full pipeline:
- Data loading and preprocessing
- MLP model definition and training
- Inference on test data
- R² score evaluation per spectral parameter

## Tech Stack

- **Python** — NumPy, scikit-learn, matplotlib
- **Version control** — GitLab (university) → GitHub

## Context

EDFA gain spectrum prediction is a key challenge in optical network planning. Accurate models allow simulation of signal behaviour across multi-span fiber links without physical measurements at every configuration.
