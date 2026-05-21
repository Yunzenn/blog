# 第 02 章：扩散模型 10 分钟速通 —— 从 DDPM 到 Flow Matching

> **论文**：[*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)
>
> **项目地址**：[ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM)
>
> **核心困惑**：Cola DLM 用的 Flow Matching 和经典 DDPM 有什么关系？为什么选择 Flow Matching 而不是 DDPM？

---

## 一、先建立直觉

想象你有一滴墨水滴入水中。前向过程是墨水逐渐扩散、最终变成均匀的浑水（噪声）。反向过程是：如果你能精确地知道每一瞬间水的流动方向（速度场），你就能倒放这个过程——从浑水恢复出那滴墨水。

- **DDPM**：把时间切成很多小段，每段学一个"去噪"操作
- **Flow Matching**：直接学习连续的速度场，一步到位

---

## 二、DDPM 回顾：离散步长的去噪

### 2.1 前向过程

DDPM 定义一个 $T$ 步的 Markov 链，逐步向数据添加高斯噪声：

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I)$$

其中 $\beta_t$ 是噪声调度（noise schedule），通常从小到大递增。

利用重参数化技巧，可以直接从 $x_0$ 跳到任意 $x_t$：

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$。

**数值例子**：假设 $\beta_t = 0.01$（常数），$T = 1000$：
- $t = 0$：$x_0$ 是原始数据
- $t = 500$：$\bar{\alpha}_{500} \approx 0.0067$，数据几乎全是噪声
- $t = 1000$：$\bar{\alpha}_{1000} \approx 0$，纯高斯噪声

### 2.2 反向过程

反向过程学习从噪声恢复数据：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

训练目标是预测噪声 $\epsilon$：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### 2.3 DDPM 的问题

1. **需要很多步**：通常 $T = 1000$ 步，推理时需要跑 1000 次模型前向（即使有 DDIM 加速也需要 50-100 步）
2. **离散步长的误差累积**：每一步的去噪都有误差，$T$ 步累积后误差可能很大
3. **方差调度需要手工设计**：$\beta_t$ 的选择对生成质量影响很大

---

## 三、Score-based 视角：连接 DDPM 和 Flow Matching 的桥梁

### 3.1 Score Function

Score function 是数据分布的对数梯度：

$$s(x) = \nabla_x \log p(x)$$

它指向数据密度增长最快的方向。如果你知道 score function，你就知道从任意点出发应该往哪个方向走才能到达高密度区域。

### 3.2 Score Matching

直接计算 $\nabla_x \log p(x)$ 需要知道 $p(x)$（这就是我们想求的），所以用一个神经网络 $s_\theta(x, t)$ 来近似。训练目标：

$$\mathcal{L}_{\text{score}} = \mathbb{E}_{t, x_0, x_t}\left[\|s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t \mid x_0)\|^2\right]$$

对于高斯噪声，$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$，所以 score matching 和 noise prediction 是等价的。

### 3.3 关键洞察

Score function 定义了一个**向量场**（vector field）。如果你知道这个向量场，就可以用 ODE（常微分方程）从噪声走到数据：

$$\frac{dx}{dt} = f(x, t) + g(t) \cdot s_\theta(x, t)$$

这就是连接 DDPM 和 Flow Matching 的桥梁——**从离散步长的去噪到连续速度场的积分**。

---

## 四、Flow Matching：直接学习连续流

### 4.1 核心思想

Flow Matching 跳过了 score function 的中间步骤，直接学习一个**速度场** $v_\psi(z_t, t)$，使得：

$$\frac{dz_t}{dt} = v_\psi(z_t, t)$$

从噪声 $z_1 \sim \mathcal{N}(0, I)$ 出发，沿着速度场积分，就能到达数据 $z_0$：

$$z_0 = z_1 + \int_1^0 v_\psi(z_t, t) \, dt = \Phi^\psi_{0 \leftarrow 1}(z_1)$$

### 4.2 条件 Flow Matching

直接学习全局速度场很难，Flow Matching 的技巧是：**对每个数据点 $z_0$，定义一个从噪声到该点的条件路径**，然后学习条件速度场。

最简单的路径是线性插值：

$$z_t = (1 - t) \cdot z_0 + t \cdot z_1, \quad t \in [0, 1]$$

对应的条件速度场：

$$u_t(z_t \mid z_0) = z_1 - z_0$$

训练目标：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, z_0, z_1}\left[\|v_\psi(z_t, t) - u_t(z_t \mid z_0)\|^2\right]$$

### 4.3 Flow Matching vs DDPM

| 维度 | DDPM | Flow Matching |
|------|------|--------------|
| 数学框架 | SDE（随机微分方程） | ODE（常微分方程） |
| "噪声"过程 | 离散步长 + 随机采样 | 连续流 + 确定性传输 |
| 训练目标 | 预测噪声 $\epsilon$ | 预测速度场 $v$ |
| 推理方式 | 多步去噪（随机） | ODE 求解（确定性） |
| 步数需求 | 通常 50-1000 步 | 通常 10-50 步 |
| 数学优雅度 | 需要精心设计 $\beta_t$ | 路径选择更自由 |

---

## 五、Cola DLM 中的 Flow Matching

### 5.1 连续流先验

Cola DLM 用 Flow Matching 参数化隐空间先验 $p_\psi(z_0)$。基础分布是标准高斯 $p_1 = \mathcal{N}(0, I)$，学习一个向量场 $v_\psi(z_t, t)$：

$$z_1 \sim \mathcal{N}(0, I), \quad \frac{dz_t}{dt} = v_\psi(z_t, t), \quad z_0 = \Phi^\psi_{0 \leftarrow 1}(z_1)$$

在代码中（`inference.py:356-357`），Euler 求解器的更新规则是：

```python
def _diffusion_dt(t_curr, t_next):
    return (float(t_curr) - float(t_next)) / max(T, 1.0)
```

每一步的更新（`inference.py:649`）：

```python
txt_next = txt - drift * dt  # z_{t-Δ} = z_t - (Δ/T) * v_ψ
```

### 5.2 时间步 schedule

Cola DLM 使用线性时间步（`inference.py:478`）：

```python
timesteps = torch.linspace(int(T), 0, timestep_num + 1, dtype=torch.float32)
```

默认 $T = 1000$，$timestep\_num = 16$，所以时间步是 $[1000, 937.5, 875, \ldots, 62.5, 0]$。

### 5.3 条件 Flow Matching 损失

训练时（代码未开源，论文式 2.1.7），损失是：

$$\mathcal{L}_{\text{FM}} = \sum_{b=1}^{B} \mathbb{E}_{t, z_0, z_1}\left[\|v_\psi(z_t^{(b)}, t; z_0^{(<b)}) - u_t^{(b)}(z_0, z_1)\|^2\right]$$

注意这里的条件：$v_\psi$ 的输入不仅有当前 block 的 $z_t^{(b)}$，还有前面所有 block 的 $z_0^{(<b)}$（stop gradient）。这就是分块因果的体现。

### 5.4 为什么选择 Flow Matching 而不是 DDPM？

1. **更少的步数**：Flow Matching 的 ODE 求解通常只需要 10-50 步，DDPM 需要 50-1000 步
2. **确定性生成**：ODE 是确定性的，相同的初始噪声总是产生相同的输出（便于调试和复现）
3. **更自然的连续空间适配**：Flow Matching 天然在连续空间定义，不需要像 DDPM 那样设计离散步长
4. **数学框架更统一**：条件 Flow Matching 的训练目标非常简洁

---

## 六、一个完整的数值例子

假设我们要用 Flow Matching 学习一个 2D 分布（两个高斯混合）：

**Step 1：采样数据点** $z_0 = (3, 2)$

**Step 2：采样噪声** $z_1 = (-0.5, 1.2)$（来自 $\mathcal{N}(0, I)$）

**Step 3：线性插值路径**，$t = 0.5$ 时：

$$z_{0.5} = (1 - 0.5) \cdot (3, 2) + 0.5 \cdot (-0.5, 1.2) = (1.25, 1.6)$$

**Step 4：条件速度场** $u_t = z_1 - z_0 = (-3.5, -0.8)$

**Step 5：训练**，让神经网络 $v_\psi(z_{0.5}, 0.5)$ 预测 $u_t = (-3.5, -0.8)$

**Step 6：推理**，从 $z_1 \sim \mathcal{N}(0, I)$ 出发，沿着学到的速度场积分 16 步，得到 $z_0$

---

## 七、面试追问清单

**基础（⭐）**：

1. DDPM 的前向过程和反向过程分别做什么？
2. Flow Matching 的训练目标是什么？
3. 为什么 Flow Matching 比 DDPM 需要更少的推理步数？

**进阶（⭐⭐）**：

4. Score function 和速度场的关系是什么？
5. 条件 Flow Matching 的"条件"是什么意思？
6. Cola DLM 中 $T = 1000$ 和 $timestep\_num = 16$ 的关系是什么？

**专家（⭐⭐⭐）**：

7. Flow Matching 和 DDPM 在什么条件下数学上等价？
8. 为什么 Cola DLM 选择线性插值路径而不是最优传输路径？
9. 分块因果的条件 Flow Matching 损失和标准 Flow Matching 损失有什么区别？

---

## 八、下期预告

下一章我们将深入离散扩散的技术细节——为什么 LLaDA 和 MDLM 选择了在离散空间做扩散，它们遇到了什么问题，以及这些问题如何推动了 Cola DLM 的设计决策。

---

> **系列导航**
>
> [第 01 章：语言生成的三次范式之争](01_generation_paradigm.md)
>
> **第 02 章：扩散模型 10 分钟速通** ← 你在这里
>
> [第 03 章：离散扩散的困境](03_discrete_diffusion.md)
>
> [第 04 章：Cola DLM 架构全景](04_cola_architecture.md)
>
> [第 05 章：Text VAE 深度解剖](05_text_vae.md)
>
> [第 06 章：分块因果 DiT 先验](06_dit_prior.md)
>
> [第 07 章：推理流水线逐行拆解](07_inference_pipeline.md)
>
> [第 08 章：工程实现评析](08_engineering.md)
>
> [第 09 章：评测复现与结果深度分析](09_benchmark.md)
>
> [第 10 章：从文本到多模态](10_future.md)

---

> **作者**：[Yunzenn](https://github.com/Yunzenn) 
