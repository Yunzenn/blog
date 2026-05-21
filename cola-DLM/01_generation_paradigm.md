# 第 01 章：语言生成的三次范式之争 —— 从 RNN 到 AR 到扩散

> **论文**：[*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)
>
> **项目地址**：[ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM)
>
> **核心困惑**：自回归（AR）Transformer 已经统治了语言模型领域，GPT、LLaMA、DeepSeek 个个都是 AR，为什么还要搞扩散语言模型？扩散模型不是用来生成图像的吗？

---

## 一、从一个类比开始

想象你在写一篇文章。自回归模型的做法是：**从第一个字开始，一个字一个字往后写**。写到第 100 个字的时候，你只能看到前面 99 个字，不知道后面要写什么。这就像一个只能"往前看"的作家——写得很流畅，但缺乏全局规划。

扩散模型的做法完全不同：**先写一个粗糙的草稿（甚至是随机噪声），然后反复修改，逐步细化**。每一轮修改都能看到整篇文章的全貌，所以天然具备全局视野。这像一个反复修改的编辑——第一稿可能很烂，但每一轮修改都在逼近最终版本。

问题是：**文本是离散的，而扩散模型天生建立在连续空间上**。这个矛盾怎么解？

---

## 二、自回归（AR）的本质：链式法则

### 2.1 数学形式

自回归模型把文本生成建模为条件概率的链式分解：

$$p(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

每一步只预测下一个 token，然后把它拼到已有序列上，继续预测。这就是 GPT 系列、LLaMA、DeepSeek 等所有主流大语言模型的工作方式。

### 2.2 为什么 AR 如此成功？

三个关键原因：

1. **训练信号密集**：每个 token 都提供一个梯度信号，数据效率高
2. **KV Cache 友好**：推理时只需计算最新 token 的 Q，历史 K/V 全部缓存复用
3. **Scaling Law 支持**：从 GPT-2 到 GPT-4，AR 的 scaling 曲线一直很稳

### 2.3 AR 的三个结构性缺陷

但 AR 有一个根本性的局限：**它只能从左到右生成**。

**缺陷 1：缺乏全局规划**

```text
AR 生成过程：
  输入: "The capital of France is"
  生成: "Paris"  ← 模型在生成 "P" 的时候就已经"决定"了答案是 Paris

但如果任务是：
  输入: "Write a poem about spring with an AABB rhyme scheme"
  生成: "The flowers bloom in morning light,"  ← 第一句就锁定了韵脚
       "The birds sing songs of pure delight,"  ← 第二句必须押韵
       "The sun warms every hill and dale,"      ← 换韵了？不，还是 AABB
       "While gentle breezes tell their tale."   ← 押韵成功
```

AR 模型在写第一句的时候就要"想好"后面三句的韵脚。虽然大模型通过海量训练数据隐式学会了这种规划，但从架构上讲，**AR 没有任何机制保证全局一致性**。

**缺陷 2：串行生成**

生成 $T$ 个 token 需要 $T$ 次前向传播。即使有 KV Cache，推理延迟仍然与输出长度线性相关。这对实时应用（聊天、代码补全）是硬伤。

**缺陷 3：Exposure Bias（曝光偏差）**

训练时，模型看到的是 ground truth 序列（teacher forcing）。推理时，模型看到的是自己的预测——如果第 5 个 token 预测错了，第 6 个 token 的输入就是错的，误差会累积。虽然现代 LLM 通过大规模预训练在一定程度上缓解了这个问题，但架构上的缺陷依然存在。

---

## 三、扩散模型的诱惑

### 3.1 图像领域的成功

扩散模型在图像生成领域取得了巨大成功。从 DDPM 到 Stable Diffusion，再到 DiT（Diffusion Transformer），图像生成质量已经达到了以假乱真的程度。

核心思想非常优雅：

$$x_0 \xrightarrow{\text{加噪}} x_1 \xrightarrow{\text{加噪}} \cdots \xrightarrow{\text{加噪}} x_T \sim \mathcal{N}(0, I)$$

前向过程逐步把数据变成噪声，反向过程学习从噪声恢复数据。每一步都能看到全局信息。

### 3.2 自然的问题：能不能在文本上做扩散？

如果扩散模型能在图像上做到这么好，为什么不能用在文本上？毕竟，语言生成也需要全局规划——写文章需要先有大纲，写代码需要先有架构设计。

但这里有一个根本矛盾：**文本是离散的，扩散是连续的**。

| | 图像 | 文本 |
|---|---|---|
| 数据空间 | 连续的像素值 | 离散的 token ID |
| "噪声"定义 | 高斯噪声叠加 | ？？？ |
| 数学工具 | Score function、ODE、SDE | ？？？ |

---

## 四、两条技术路线的分叉

面对这个矛盾，研究者们走了两条不同的路：

### 4.1 路线 A：离散扩散（MDLM、LLaDA、Plaid）

**核心思路**：既然文本是离散的，那就设计离散的扩散过程。

典型做法是 **mask-and-predict**：
1. 随机 mask 掉一部分 token（类比连续扩散的"加噪"）
2. 训练模型预测被 mask 的 token（类比"去噪"）
3. 推理时从全 mask 状态开始，逐步预测出所有 token

数学上，用离散的 Markov 链替代连续的 SDE/ODE：

$$q(x_t | x_0) = \text{Categorical}(\alpha_t x_0 + (1 - \alpha_t) \text{[MASK]})$$

**优点**：
- 直接在 token 空间操作，不需要额外的编码/解码
- 实现相对简单
- LLaDA 已经证明了可行性（8B 参数，不错的 benchmark 成绩）

**缺点**：
- "噪声"定义不自然：mask 是一个离散的 0/1 跳变，不是连续退化
- 数学工具受限：离散空间没有连续的 score function，无法直接套用 Flow Matching
- 信息瓶颈：被 mask 的 token 信息完全丢失（不像连续噪声那样保留部分信息）

### 4.2 路线 B：连续隐空间扩散（Cola DLM）

**核心思路**：既然问题出在离散空间，那就离开它——用 VAE 把文本映射到连续空间，然后在连续空间里做扩散。

$$\text{离散 token } x \xrightarrow{\text{VAE encoder}} \text{连续隐变量 } z_0 \xrightarrow{\text{扩散}} \text{生成 } z_0 \xrightarrow{\text{VAE decoder}} \text{离散 token } x$$

**优点**：
- 扩散过程回归连续空间，Flow Matching 等数学工具可以直接使用
- "噪声"定义自然：标准高斯 $\mathcal{N}(0, I)$
- 隐空间可以做平滑插值（token 空间做不到）
- 天然支持多模态扩展（不同模态共享同一个隐空间先验）

**缺点**：
- 引入了 VAE 这个额外组件，训练更复杂
- VAE 的隐空间质量成为整个系统的瓶颈
- 三层架构（VAE + DiT + Decoder）比端到端 AR 更难优化

---

## 五、全景对比

| 维度 | AR（GPT/LLaMA） | 离散扩散（LLaDA） | 连续隐空间扩散（Cola DLM） |
|------|-----------------|-------------------|--------------------------|
| 数学形式 | $p(x) = \prod p(x_t \mid x_{<t})$ | 离散 Markov chain + mask | VAE + Flow Matching ODE |
| 生成方向 | 左→右，逐 token | 全局→局部，逐步 unmask | 块级，先验传输 |
| 全局规划 | 架构上不支持 | 天然支持（每步看全局） | 天然支持（隐空间全局） |
| 推理速度 | $T$ 步（可并行化有限） | $S$ 步（$S$ 通常 < $T$） | $B \times S$ 步（$B$ = block 数） |
| 训练信号 | 每 token 一个 | 每 unmask 步一个 | VAE 重构 + Flow Matching |
| 数学优雅度 | 高（链式法则） | 中（离散版本的 score matching） | 高（连续 ODE） |
| 多模态扩展 | 需要特殊设计 | 需要离散化各模态 | 天然支持（各模态独立 VAE） |
| 成熟度 | 极高（GPT-4 级别） | 中（LLaDA 8B） | 低（Cola DLM 2B） |

---

## 六、Cola DLM 的核心洞察

Cola DLM 的论文（arXiv:2605.06548）提出了一个关键洞察：

> **扩散过程做的是隐空间先验传输（latent prior transport），而不是 token 级观测恢复。**

这句话的意思是：扩散模型在 Cola DLM 中不是直接生成文本（像 LLaDA 那样），而是在隐空间里学习一个从噪声到"有意义的隐向量"的映射。文本的生成由 VAE decoder 负责。

用我们的类比来说：
- LLaDA（离散扩散）：直接在"人类语言"里做反复修改
- Cola DLM（连续隐空间扩散）：先在"世界语"里写好草稿，再翻译回人类语言

这个设计把"全局语义组织"（由 DiT 先验负责）和"局部文本实现"（由 VAE decoder 负责）**显式解耦**了。

### 6.1 层次化联合分布

Cola DLM 的数学框架是一个层次化隐变量模型：

$$p(x, z_0) = p_\theta(x \mid z_0) \cdot p_\psi(z_0)$$

其中：
- $q_\phi(z_0 \mid x)$：VAE 编码器，把文本映射到隐空间（**不属于生成模型本身**，只用于推断）
- $p_\psi(z_0)$：DiT 学习的隐空间先验（**这才是生成模型的核心**）
- $p_\theta(x \mid z_0)$：VAE 解码器，把隐变量还原为文本

### 6.2 分块因果分解

为了处理序列数据，Cola DLM 把隐变量沿序列维分成 $B$ 个 block：

$$p_\psi(z_0) = p_\psi(z_0^{(1)}) \cdot \prod_{b \geq 2} p_\psi(z_0^{(b)} \mid z_0^{(<b)})$$

- **block 间因果**：第 $b$ 个 block 只能看到前面的 block（类似 AR）
- **block 内双向**：同一个 block 内的所有位置互相可见（类似 BERT）

这是一种介于纯 AR（逐 token 因果）和全双向（所有位置互相可见）之间的折中设计。

---

## 七、2026 年的批判性视角

### 7.1 "扩散 vs AR"的边界比想象的模糊

Cola DLM 的分块因果结构在 block 间引入了因果顺序——第 2 个 block 必须等第 1 个 block 生成完才能开始。这本质上是一种**粗粒度的自回归**。

那么问题来了：如果 block_size=1，分块因果就退化为逐 token 的 AR；如果 block_size=序列长度，分块因果就退化为全双向。Cola DLM 的 block_size=4，这是一个经验值，最优值取决于任务特性。

### 7.2 真正的优势在哪里？

扩散语言模型相比 AR 的真正优势，可能不在于"全局规划"（大模型通过 chain-of-thought 也能做到），而在于：

1. **并行生成**：同一个 block 内的所有 token 是并行生成的，不是逐 token 的
2. **多模态统一**：连续隐空间天然支持不同模态的联合建模
3. **理论优雅**：Flow Matching 的数学框架比 AR 的链式法则更统一

### 7.3 真正的瓶颈在哪里？

Cola DLM 的当前瓶颈不是扩散本身，而是 **VAE 的隐空间质量**。如果 VAE 编码不好（丢失语义、隐空间不规整），再强的 DiT 先验也救不了。论文报告的 2B 参数模型 Task Average 26.75%，与同规模 AR 模型差距明显，说明 VAE 的隐空间还有很大的提升空间。

---

## 八、面试追问清单

**基础（⭐）**：

1. 自回归模型的链式分解公式是什么？为什么它天然适合文本生成？
2. 扩散模型的前向过程和反向过程分别做什么？
3. 离散扩散和连续扩散的核心区别是什么？

**进阶（⭐⭐）**：

4. Cola DLM 的分块因果注意力和标准 causal attention 有什么区别？
5. 为什么 Cola DLM 要用 VAE 而不是直接在 token 空间做扩散？
6. "隐空间先验传输"和"token 级观测恢复"有什么区别？

**专家（⭐⭐⭐）**：

7. 如果 block_size=1，Cola DLM 的分块因果结构会退化成什么？如果 block_size=序列长度呢？
8. Cola DLM 的三层解耦（VAE + DiT + Decoder）和 Stable Diffusion 的三层解耦（VAE + UNet + Decoder）有什么异同？
9. 扩散语言模型能否 scale 到 AR 模型的水平？主要挑战是什么？

---

## 九、下期预告

下一章我们将深入扩散模型的数学基础——从 DDPM 到 Flow Matching。你会看到 Cola DLM 为什么选择 Flow Matching 而不是经典的 DDPM，以及"连续流"这个概念到底意味着什么。

---

> **系列导航**
>
> **第 01 章：语言生成的三次范式之争** ← 你在这里
>
> [第 02 章：扩散模型 10 分钟速通 —— 从 DDPM 到 Flow Matching](02_diffusion_foundation.md)
>
> [第 03 章：离散扩散的困境 —— 为什么 Mask-and-Predict 走不远？](03_discrete_diffusion.md)
>
> [第 04 章：Cola DLM 架构全景 —— 三层解耦的设计哲学](04_cola_architecture.md)
>
> [第 05 章：Text VAE 深度解剖 —— 文本 ↔ 隐空间的双向映射](05_text_vae.md)
>
> [第 06 章：分块因果 DiT 先验 —— 在隐空间里做 Flow Matching](06_dit_prior.md)
>
> [第 07 章：推理流水线逐行拆解 —— 从 prompt 到生成文本](07_inference_pipeline.md)
>
> [第 08 章：工程实现评析 —— 优秀实践与改进空间](08_engineering.md)
>
> [第 09 章：评测复现与结果深度分析](09_benchmark.md)
>
> [第 10 章：从文本到多模态 —— 统一生成的未来](10_future.md)

---

> **作者**：[Yunzenn](https://github.com/Yunzenn) 
