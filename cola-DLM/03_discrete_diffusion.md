# 第 03 章：离散扩散的困境 —— 为什么 Mask-and-Predict 走不远？

> **论文**：[*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)
>
> **项目地址**：[ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM)
>
> **核心困惑**：LLaDA、MDLM 已经证明了离散扩散能工作（LLaDA 8B 在多个 benchmark 上表现不错），Cola DLM 为什么还要"绕路"去连续空间？

---

## 一、离散扩散的基本思路

离散扩散的思路很直接：**既然文本是离散的 token，那就设计离散的"噪声"过程**。

最主流的做法是 **mask-and-predict**（也叫 absorbing-state diffusion）：

1. **前向过程**：随机把一部分 token 替换为 `[MASK]`
2. **反向过程**：训练模型预测被 mask 的 token
3. **推理**：从全 `[MASK]` 状态开始，逐步预测出所有 token

数学上，前向过程是一个离散 Markov 链：

$$q(x_t \mid x_0) = \text{Categorical}(\alpha_t \cdot x_0 + (1 - \alpha_t) \cdot \text{[MASK]})$$

其中 $\alpha_t$ 从 1 递减到 0，控制 mask 比例。

---

## 二、三个根本问题

### 问题 1："噪声"定义不自然

连续扩散的噪声是高斯噪声——一个连续的、可微的、有丰富数学结构的对象。你可以叠加、插值、做 score matching。

离散扩散的"噪声"是 mask——一个离散的 0/1 跳变：

```text
原始:  "The cat sat on the mat"
mask:  "The [M] [M] on the [M]"   ← 部分 token 被替换
全mask: "[M] [M] [M] [M] [M] [M]" ← 所有 token 被替换
```

这带来几个问题：

- **信息是二值的**：一个 token 要么被完全保留，要么被完全丢失。不像连续噪声那样可以"部分退化"。
- **没有自然的"中间状态"**：$\alpha_t = 0.5$ 意味着 50% 的 token 被 mask，但被 mask 的 token 之间没有"程度"差异。
- **不可微**：mask 操作是离散的，无法直接用梯度下降优化。

### 问题 2：数学工具受限

连续扩散有一整套优美的数学工具：

| 工具 | 连续扩散 | 离散扩散 |
|------|---------|---------|
| Score function | $\nabla_x \log p(x)$ | 离散版本存在，但形式复杂 |
| ODE/SDE | $\frac{dx}{dt} = f(x,t) + g(t) \cdot s(x,t)$ | 不适用 |
| Flow Matching | 直接学习速度场 | 不适用 |
| 最优传输 | 有高效的算法 | 需要特殊适配 |

离散空间没有连续的梯度，所以 score function、ODE、Flow Matching 这些工具都无法直接使用。虽然有离散版本的 score matching（如 MDLM 的做法），但形式更复杂，且不如连续版本优雅。

### 问题 3：信息瓶颈

考虑一个简单的例子：

```text
原始句子: "The quick brown fox jumps over the lazy dog"
mask 后:  "The [M] [M] fox [M] over the [M] dog"
```

模型需要从 `[M]` 的位置预测出 "quick"、"brown"、"jumps"、"lazy"。但 `[M]` 不携带任何信息——它只是一个占位符。模型完全依赖上下文（"The ... fox ... over the ... dog"）来推断。

相比之下，连续扩散中的"噪声"是这样的：

```text
原始向量: [0.82, -0.31, 0.45, 0.12, -0.67]
噪声后:   [0.71, -0.18, 0.38, 0.05, -0.54]  ← 信息被"模糊化"而非"擦除"
```

连续噪声保留了部分信息——向量的方向大致不变，只是幅度被衰减。这给了模型更多的"线索"来恢复原始数据。

---

## 三、LLaDA/Plaid 的 Workaround 及其代价

### 3.1 LLaDA 的做法

LLaDA（Large Language Diffusion with mAsking）是目前最成功的离散扩散语言模型之一。它的核心设计：

1. **预训练**：在大规模语料上做 mask-and-predict，mask 率随机采样
2. **推理**：从全 mask 状态开始，按 mask 率从高到低逐步预测
3. **SFT**：用指令数据微调，支持多轮对话

### 3.2 需要的 Workaround

为了让离散扩散工作，LLaDA 需要几个 workaround：

**Workaround 1：精心设计的 mask 率调度**

不能简单地线性降低 mask 率。LLaDA 发现需要：
- 训练时随机采样 mask 率（而不是固定 schedule）
- 推理时使用非均匀的 schedule（某些步的 mask 率变化更大）

**Workaround 2：多步预测**

每次 unmask 时不是只预测一个 token，而是同时预测所有被 mask 的位置。这需要模型有很强的全局推理能力。

**Workaround 3：半自回归推理**

为了提高生成质量，LLaDA 在推理时引入了"半自回归"策略：先生成第一个 block，再生成第二个 block，以此类推。这本质上是把离散扩散和 AR 混合了。

### 3.3 代价

这些 workaround 带来了额外的复杂性：

- mask 率调度需要大量实验调优
- 多步预测的训练目标和推理策略不一致
- 半自回归推理削弱了"全局生成"的理论优势

---

## 四、Cola DLM 的洞察：离开离散空间

Cola DLM 的论文提出了一个根本性的洞察：

> **如果问题出在离散空间，那就离开它。**

具体做法：

$$\text{离散 token } x \xrightarrow{\text{VAE encoder } q_\phi} \text{连续隐变量 } z_0 \xrightarrow{\text{Flow Matching}} \text{生成 } z_0 \xrightarrow{\text{VAE decoder } p_\theta} \text{离散 token } x$$

这个设计把问题拆成了两个子问题：

1. **映射问题**：如何把离散文本映射到连续空间？→ VAE 负责
2. **生成问题**：如何在连续空间生成有意义的向量？→ Flow Matching 负责

每个子问题都在它最适合的空间里解决。

---

## 五、对比表：离散扩散 vs 连续隐空间扩散

| 维度 | 离散扩散（LLaDA） | 连续隐空间扩散（Cola DLM） |
|------|-------------------|--------------------------|
| "噪声"定义 | `[MASK]` token（离散跳变） | 高斯噪声 $\mathcal{N}(0, I)$（连续退化） |
| 数学工具 | 离散 score matching | Flow Matching ODE |
| 信息保留 | 完全丢失（被 mask 的 token） | 部分保留（向量方向大致不变） |
| 训练目标 | 预测被 mask 的 token | 预测速度场 $v_\psi$ |
| 推理方式 | 逐步 unmask | ODE 积分 |
| 额外组件 | 无（直接在 token 空间） | 需要 VAE（encoder + decoder） |
| 训练复杂度 | 低（单阶段） | 高（两阶段：VAE 预训练 + 联合训练） |
| 多模态扩展 | 需要离散化各模态 | 天然支持（各模态独立 VAE） |
| 数学优雅度 | 中 | 高 |

---

## 六、一个思想实验

假设我们要生成一句话："The capital of France is Paris"。

**离散扩散的做法**：

```text
Step 1: [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
Step 2: The    [MASK] [MASK] [MASK] is    [MASK]
Step 3: The    capital of    [MASK] is    [MASK]
Step 4: The    capital of    France is    Paris
```

每一步，模型需要从上下文推断被 mask 的 token。问题是：在 Step 2，模型怎么知道 `[MASK][MASK][MASK]` 应该是 "capital of France"？它只能靠上下文猜测。

**连续隐空间扩散的做法**：

```text
Step 1: z_1 ~ N(0, I)  → 随机噪声向量
Step 2: z_{0.8}         → 语义模糊但大致可辨的向量
Step 3: z_{0.4}         → 语义清晰的向量
Step 4: z_0             → 精确的隐向量 → decoder → "The capital of France is Paris"
```

每一步，隐向量都在"平滑地"从噪声变成有意义的表示。模型看到的是一个连续变化的"语义场"，而不是离散的"有/无"。

---

## 七、2026 年的批判性视角

### 7.1 离散扩散并没有"失败"

需要强调：离散扩散并不是一个失败的方向。LLaDA 8B 在多个 benchmark 上取得了不错的结果，证明了离散扩散的可行性。Cola DLM 的论文也承认，离散扩散是"一个有效的替代方案"。

### 7.2 连续隐空间扩散的真正优势

Cola DLM 的优势不在于"离散扩散不行"，而在于：

1. **数学框架更统一**：Flow Matching 是一个通用框架，可以自然地扩展到多模态
2. **隐空间的可操作性**：连续隐空间可以做插值、组合、条件生成等操作
3. **理论上的表达能力**：连续 ODE 比离散 Markov chain 有更强的表达能力

### 7.3 真正的瓶颈

Cola DLM 当前的瓶颈不是"连续 vs 离散"的选择，而是 **VAE 的隐空间质量**。如果 VAE 编码不好，连续空间的优势就无法发挥。论文报告的 2B 参数模型 Task Average 26.75%，与 LLaDA 8B 的差距主要来自模型规模和训练数据，而非范式选择。

---

## 八、面试追问清单

**基础（⭐）**：

1. 离散扩散的 mask-and-predict 过程是什么？
2. 为什么离散扩散不能直接使用 Flow Matching？
3. Cola DLM 用什么来解决"文本是离散的"这个问题？

**进阶（⭐⭐）**：

4. 离散扩散的"信息瓶颈"问题是什么？如何缓解？
5. LLaDA 的"半自回归"推理策略是怎么回事？
6. VAE 的隐空间质量如何影响 Cola DLM 的生成质量？

**专家（⭐⭐⭐）**：

7. 离散 score matching 和连续 score matching 的数学关系是什么？
8. 如果 VAE 的隐空间是完美的（无损编码），Cola DLM 的生成质量上限是什么？
9. 连续隐空间扩散能否和 RLHF 结合？主要挑战是什么？

---

## 九、下期预告

前三章我们理解了"为什么要这样做"，从下一章开始进入"具体怎么做"。我们将详细拆解 Cola DLM 的三层架构——Text VAE、分块因果 DiT、条件解码器——各自负责什么，如何协作。

---

> **系列导航**
>
> [第 01 章：语言生成的三次范式之争](01_generation_paradigm.md)
>
> [第 02 章：扩散模型 10 分钟速通](02_diffusion_foundation.md)
>
> **第 03 章：离散扩散的困境** ← 你在这里
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
