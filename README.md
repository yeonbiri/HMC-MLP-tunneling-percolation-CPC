# Hybrid Monte Carlo–MLP Prediction of Tunneling Conductivity in CPCs

This repository contains the MATLAB implementation of a hybrid computational framework for predicting the electrical behavior of conductive polymer composites (CPCs) under compressive strain.

## 1. Overview
The study integrates **3D Monte Carlo (MC) simulations** with a **Multilayer Perceptron (MLP)** surrogate model. The MC simulation generates high-fidelity stochastic data on percolation networks, which is then used to train an MLP for near-instantaneous prediction of the Infinite Cluster Ratio (ICR).



[Image of a neural network architecture diagram]


## 2. Code Descriptions

### [1] `Simulationcutoff.m` (Data Generation)
This script performs stochastic 3D Monte Carlo simulations to evaluate percolation behavior.
- **Physics Engine:** Models spherical fillers in a Representative Volume Element (RVE) with tunneling-driven connectivity (Soft-shell model).
- **Parameters:**
  - **Poisson's Ratio:** 0.0, 0.3, 0.5
  - **Volume Fractions:** Variable (up to 800 units)
  - **Filler Sizes:** 0.1 to 5.0
  - **Tunneling Cutoff ($\delta$):** 0.05, 0.1, 0.2 nm
- **Output:** Generates `.mat` and `.xlsx` files containing mean cluster counts and ICR for 9 compressive strain steps across 100 repetitions per combination.

### [2] `MLPmodel.m` (Surrogate Modeling)
This script builds and validates the machine learning surrogate model.
- **Architecture:** 3-layer MLP [50, 30, 15] with ReLU activation.
- **Features:** Integrates datasets from different tunneling cutoffs, performs data normalization, and utilizes L2 regularization (Lambda = 0.001).
- **Validation:** Includes a **5-Fold Cross-Validation** to ensure model robustness and stability (typically achieving $R^2 > 95\%$).

## 3. Getting Started

### Prerequisites
- **MATLAB** (R2021a or later)
- **Statistics and Machine Learning Toolbox** (Required for `fitrnet`)

### Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/yeonbiri/HMC-MLP-tunneling-percolation-CPC.git](https://github.com/yeonbiri/HMC-MLP-tunneling-percolation-CPC.git)
