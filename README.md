# Managing a Portfolio Using the GRPO Approach

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![Reinforcement Learning](https://img.shields.io/badge/RL-GRPO-green)](https://arxiv.org/abs/2209.01814)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)](https://numpy.org/)

---

## 📌 **Project Overview**

This repository implements **Group Relative Policy Optimization (GRPO)** for **portfolio management** in financial markets using **synthetic data**. The goal is to train an agent to make effective trading decisions based solely on historical asset variations, without access to future information. The model leverages an **LSTM-based architecture** to remember past states through a memory vector, enabling it to capture temporal dependencies in the data.

---

## 🚀 **Key Features**

### 1. **GRPO Algorithm**
A modern reinforcement learning method that optimizes policies using relative advantages within groups of trajectories.

- **Relative Advantage Calculation:**
  Rewards are normalized within groups of trajectories sharing the same initial state to estimate advantages. The relative advantage \( A(s,a) \) is computed as:
  \[
  A(s,a) = \frac{R(s,a) - \mu_R(s)}{\sigma_R(s)}
  \]
  where \( R(s,a) \) is the reward for action \( a \) in state \( s \), and \( \mu_R(s) \), \( \sigma_R(s) \) are the mean and standard deviation of rewards for trajectories starting from \( s \).

- **LSTM-Based Policy:**
  The model uses an **LSTM** to encode temporal dependencies in asset variations, allowing it to retain a memory of past states. The LSTM cell update is defined by:
  \[
  \begin{align*}
  f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)} \\
  i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)} \\
  \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(Candidate Memory)} \\
  C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State Update)} \\
  o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)} \\
  h_t &= o_t \odot \tanh(C_t) \quad \text{(Hidden State)}
  \end{align*}
  \]

- **Multi-File Buffer:**
  Efficient memory management for batch training on collected transitions.

---

### 2. **Trading Environment**
- **Synthetic Data:**
  The agent is trained and evaluated on **synthetic asset variation data**, simulating realistic market conditions.

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
 | **`grpo_var_data.py`** | Main script implementing the GRPO algorithm with LSTM for processing asset variation data. Includes training loops, policy evaluation, and portfolio performance tracking. |

---
## 📊 **Performance Analysis**

### **Training Results**
- The model's portfolio evolution **outperforms a buy-and-hold strategy**, indicating that it successfully captures temporal dependencies in the synthetic asset variations.
- The training curve shows consistent improvement, suggesting that the LSTM-based memory mechanism helps the model adapt to past states and refine its policy over time.

### **Challenges and Next Steps**
- **High Noise Levels:**
  The weak linear dependencies in the synthetic asset variations make it challenging for the model to extract clear signals. The LSTM's memory capability helps mitigate this by retaining relevant past information.
- **Future Work:**
  Integrate low-accuracy asset predictions to provide the model with additional signals, potentially improving its decision-making process.

---
## 🛠️ **Future Improvements**
- **KL Divergence Constraint:**
  Introduce a stricter Kullback-Leibler divergence term to prevent abrupt policy updates:
  \[
  \mathcal{L}_{KL} = \beta \cdot D_{KL}(\pi_{\text{new}} \| \pi_{\text{old}})
  \]

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
