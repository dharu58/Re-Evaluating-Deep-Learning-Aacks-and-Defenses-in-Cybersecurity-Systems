# Re-Evaluating Deep Learning Attacks and Defenses in Cybersecurity Systems (Replication)

This repository contains a full replication and operationalization of the adversarial AI framework detailed in the 2024 paper: *"Re-Evaluating Deep Learning Attacks and Defenses in Cybersecurity Systems"* (Ahmed et al.). 

In addition to replicating the backend attack and defense pipelines, this project features a interactive web application interface. Users can upload a target network traffic dataset, evaluate its vulnerability under various advanced adversarial threat models, and observe the defensive performance gains achieved via localized adversarial retraining.

## 🚀 Project Features

- **Multi-Dataset Preprocessing Engine:** Standardized loading wrappers for `ADFA-LD`, `CSE-CICIDS2018`, and `CSE-CICIDS2019`.
- **Four Adversarial Attack Modules:**
  - **DeepFool:** White-box, decision boundary-targeted minimal perturbation generation.
  - **Kernel Density Estimation (KDE):** Non-parametric statistical probability density manipulation.
  - **Zeroth-Order Optimization (ZOO):** Derivative-free black-box gradient estimation.
  - **Generative Adversarial Networks (GAN):** Co-trained synthetic adversarial traffic injection.
- **Robust Defense Implementation:** Defensive classification leveraging iterative DeepFool-based adversarial training.
- **Interactive Web UI:** Front-end dashboard to upload network logs, simulate arbitrary attack ratios, and view real-time accuracy/f1-score drop-off maps.

## 🛠️ Architecture Overview


## 📋 Installation & Setup

### Backend (Core ML Engine)
1. Navigate to the core engine:
   ```bash
   cd core_engine
