# Managing a Portfolio Using the GRPO Approach

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![Reinforcement Learning](https://img.shields.io/badge/RL-GRPO-green)](https://arxiv.org/abs/2209.01814)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)](https://numpy.org/)

---

## 📌 **Project Overview**

This repository implements **Group Relative Policy Optimization (GRPO)** for **portfolio management** in financial markets using **synthetic data**. The synthetic data includes **intentional linear relationships between assets**, deliberately embedded in high levels of noise. The goal is to evaluate whether the model can detect these relationships to improve its portfolio management decisions.

The model leverages an **LSTM-based architecture** to remember past states through a memory vector, enabling it to capture temporal dependencies in the data and potentially uncover the underlying linear relationships despite the noise.

---

## 🚀 **Key Features**

### 1. **GRPO Algorithm**
A modern reinforcement learning method that optimizes policies using relative advantages within groups of trajectories.

- **Relative Advantage Calculation:**
  Rewards are normalized within groups of trajectories sharing the same initial state to estimate advantages.
  ![Relative Advantage Equation](https://latex.codecogs.com/svg.latex?A%28s%2Ca%29%20%3D%20%5Cfrac%7BR%28s%2Ca%29%20-%20%5Cmu_R%28s%29%7D%7B%5Csigma_R%28s%29%7D)
  where \( R(s,a) \) is the reward for action \( a \) in state \( s \), and \( \mu_R(s) \), \( \sigma_R(s) \) are the mean and standard deviation of rewards for trajectories starting from \( s \).

- **LSTM-Based Policy:**
  The model uses an **LSTM** to encode temporal dependencies in asset variations, allowing it to retain a memory of past states. The LSTM cell update is defined by:
  ![LSTM Equations](https://latex.codecogs.com/svg.latex?%5Cbegin%7Baligned%7D%20f_t%20%26%3D%20%5Csigma%28W_f%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_f%29%20%5Cquad%20%5Ctext%7B%28Forget%20Gate%29%7D%20%5C%5C%20i_t%20%26%3D%20%5Csigma%28W_i%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_i%29%20%5Cquad%20%5Ctext%7B%28Input%20Gate%29%7D%20%5C%5C%20%5Ctilde%7BC%7D_t%20%26%3D%20%5Ctanh%28W_C%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_C%29%20%5Cquad%20%5Ctext%7B%28Candidate%20Memory%29%7D%20%5C%5C%20C_t%20%26%3D%20f_t%20%5Codot%20C_%7Bt-1%7D%20%2B%20i_t%20%5Codot%20%5Ctilde%7BC%7D_t%20%5Cquad%20%5Ctext%7B%28Cell%20State%20Update%29%7D%20%5C%5C%20o_t%20%26%3D%20%5Csigma%28W_o%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_o%29%20%5Cquad%20%5Ctext%7B%28Output%20Gate%29%7D%20%5C%5C%20h_t%20%26%3D%20o_t%20%5Codot%20%5Ctanh%28C_t%29%20%5Cquad%20%5Ctext%7B%28Hidden%20State%29%7D%20%5Cend%7Baligned%7D)

- **Multi-File Buffer:**
  Efficient memory management for batch training on collected transitions.

---

### 2. **Trading Environment**
- **Synthetic Data with Linear Relationships:**
  The agent is trained and evaluated on **synthetic asset variation data**, where **linear relationships between assets are intentionally introduced and embedded in high noise levels**. The goal is to test whether the model can detect these relationships to improve its portfolio management.

- **Input Data:**
  The agent receives the 8 most recent historical variations for each asset, formatted as a tensor of shape \( (m, n, p) \), where:
  - \( m \): Batch size
  - \( n \): Number of assets
  - \( p \): Number of historical variations per asset

- **Policy Output:**
  The model outputs a policy \( \pi(a s) \) to allocate assets, with the action space discretized into \( (2 \times \text{flexibility} + 1) \) possible allocations per asset.

---
## 📂 **Repository Structure**
 | File | Description |
 |------|-------------|
 | **`main.py`** | Main script implementing the GRPO algorithm with LSTM for processing asset variation data. Includes training loops, policy evaluation, and portfolio performance tracking. |
 | **`data.py`** | Script for generating and managing synthetic asset variation data, including linear relationships and noise. |
 | **`plot.py`** | Utility script for visualizing training progress, portfolio performance, and other metrics. |

---
## 📊 **Performance Analysis**

### **Training Results**
- The model's portfolio evolution **outperforms a buy-and-hold strategy**, indicating that it successfully captures temporal dependencies in the synthetic asset variations.
- The training curve shows consistent improvement, suggesting that the LSTM-based memory mechanism helps the model adapt to past states and refine its policy over time.
- **Detection of Linear Relationships:**
  The model's ability to outperform the baseline suggests it may be detecting the underlying linear relationships between assets, despite the high noise levels.

### **Challenges and Next Steps**
- **High Noise Levels:**
  The weak linear dependencies in the synthetic asset variations make it challenging for the model to extract clear signals. The LSTM's memory capability helps mitigate this by retaining relevant past information.
- **Future Work:**
  Further experiments could involve adjusting the noise levels or introducing more complex relationships to test the model's robustness and ability to detect patterns.

---
## 🛠️ **Future Improvements**
- **KL Divergence Constraint:**
  Introduce a stricter Kullback-Leibler divergence term to prevent abrupt policy updates:
  ![KL Divergence Equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D_%7BKL%7D%20%3D%20%5Cbeta%20%5Ccdot%20D_%7BKL%7D%28%5Cpi_%7B%5Ctext%7Bnew%7D%7D%20%5C%7C%5C%7C%20%5Cpi_%7B%5Ctext%7Bold%7D%7D%29)

- **Prioritized Experience Replay (PER):**
  Prioritize successful trajectories in the buffer to reinforce learned behaviors.

- **Learning Rate Scheduling:**
  Use `ReduceLROnPlateau` to dynamically adjust the learning rate based on performance plateaus.

- **Architecture Enhancements:**
  Explore **Transformer-based models** to better capture long-term dependencies in asset variations.

---
## 🚀 **Installation & Usage**

### **Prerequisites**
- Python 3.9+
- PyTorch
- NumPy
- Matplotlib (for visualization)

Install the required packages:
```bash
pip install torch numpy matplotlib
