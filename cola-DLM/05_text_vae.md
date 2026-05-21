# 第 05 章：Text VAE 深度解剖 —— 文本 ↔ 隐空间的双向映射

> **论文**：[*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)
> **项目地址**：[ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM)
> **源码**：[`modeling_cola_vae.py`](https://github.com/ByteDance-Seed/Cola-DLM/blob/main/cola_dlm/modeling_cola_vae.py)
>
> **核心困惑**：Text VAE 是怎么把离散 token 序列变成连续向量的？隐空间长什么样？

---

## 一、VAE 基础回顾

### 1.1 经典 VAE

VAE（Variational Autoencoder）的核心思想：

$$\text{Encoder: } x \xrightarrow{q_\phi(z|x)} z \xrightarrow{p_\theta(x|z)} \text{Decoder: } \hat{x}$$

训练目标（ELBO）：

$$\mathcal{L} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \beta \cdot \text{KL}(q_\phi(z|x) \| p(z))$$

第一项是重构损失（让解码器能还原输入），第二项是 KL 正则（让隐空间规整）。

### 1.2 Cola Text VAE 的特殊之处

Cola 的 Text VAE 不是经典的 CNN-based VAE，而是一个 **Transformer-based VAE**：

- Encoder：token embedding → Conv1d patchify → Transformer blocks → 线性投影 → 隐向量
- Decoder：隐向量 → 线性投影 → Transformer blocks → unpatch → token logits

---

## 二、编码器：从 token 到隐向量

### 2.1 Patch Embedding（Conv1d）

代码位置：`modeling_cola_vae.py:514`

```python
patch_embedder=nn.Conv1d(config.dim, config.dim,
                         kernel_size=config.patch_size,
                         stride=config.patch_size)
```

作用：把 token 序列压缩。假设 `patch_size=1`，则每个 token 对应一个隐向量；如果 `patch_size=4`，则每 4 个 token 压缩为 1 个隐向量。

每个样本单独处理（因为 Conv1d 需要固定长度）：

```python
# modeling_cola_vae.py:563-578
def _encode_patch_per_sample(self, input_ids_list):
    out_list = []
    for ids in input_ids_list:
        x = self.encoder.wte(ids.unsqueeze(0))  # (1, L_i, d)
        x = x.permute(0, 2, 1)                   # (1, d, L_i)
        x = self.encoder.patch_embedder(x)        # (1, d, n_i)
        x = x.permute(0, 2, 1).squeeze(0)        # (n_i, d)
        out_list.append(x)
    return out_list
```

### 2.2 Transformer Encoder Blocks

代码位置：`modeling_cola_vae.py:515`

```python
blocks=nn.ModuleList([TextVAEBlock(**block_kwargs)
                      for _ in range(config.encoder_num_blocks)])
```

默认配置：4 个 encoder blocks（`configuration_cola_vae.py:71`）。

每个 `TextVAEBlock`（`modeling_cola_vae.py:260-462`）包含：

| 组件 | 实现 | 作用 |
|------|------|------|
| Norm | LayerNorm（非 RMSNorm），默认 `post_norm=True` | 稳定训练（见下文详解） |
| QKV projection | `nn.Linear(dim, 3*dim)`（`shared_heads_kv=1` 时） | 计算 Q, K, V |
| QK-norm | `nn.LayerNorm(head_dim)` | 防止注意力 logits 爆炸 |
| RoPE | `VAERotaryEmbedding` | 位置编码（`rope_theta=500000`） |
| Attention | `slow_attn` | 标准 softmax attention |
| FFN | SwiGLU（`F.silu(gate) * x`） | 非线性变换 |

**Norm 放置方式**：默认 `post_norm=True`，这是一种非标准的 norm 模式——attention 路径为 `LN(x) + Attn(x)`（对原始输入做 LN，然后加残差），FFN 路径为 `residual + LN(FFN(x))`（FFN 后做 LN）。既不是标准 pre-norm（`x + Attn(LN(x))`），也不是标准 post-norm（`LN(x + Attn(x))`）。

**注意**：VAE 用 **SwiGLU** 激活函数，而 DiT 用 **GELU tanh**。这是一个设计选择差异。

### 2.3 最终投影

代码位置：`modeling_cola_vae.py:523-526`

如果 `use_variation=True`（默认）：

```python
encoder_dict["final_layer"] = nn.Linear(config.dim, config.latent_dim * 2)
```

输出维度是 `latent_dim * 2`，拆成 mean 和 logvar：

```python
# modeling_cola_vae.py:80
self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
```

### 2.4 DiagonalGaussianDistribution

代码位置：`modeling_cola_vae.py:76-100`

```python
class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)  # 防止数值溢出
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator=None):
        sample = torch.randn(self.mean.shape, ...)
        return self.mean + self.std * sample  # 重参数化技巧

    def mode(self):
        return self.mean  # 推理时直接用 mean
```

---

## 三、解码器：从隐向量到 token

### 3.1 架构

代码位置：`modeling_cola_vae.py:536-544`

```python
self.decoder = nn.ModuleDict(dict(
    in_layer=nn.Linear(config.latent_dim, config.dim),  # 隐空间 → 模型维度
    blocks=nn.ModuleList([TextVAEBlock(**block_kwargs)
                          for _ in range(config.decoder_num_blocks)]),  # 4 blocks
    unpatch_layer=nn.Linear(config.dim, config.patch_size * config.dim),  # unpatch
    final_norm=build_norm_layer(...),
    final_layer=nn.Linear(config.dim, config.vocab_size),  # → vocab logits
))
```

### 3.2 Unpatch 操作

代码位置：`modeling_cola_vae.py:687-688`

```python
z = self.decoder.unpatch_layer(z)  # (1, L_q, d*patch_size)
z = rearrange(z, "b l (c ps) -> b (l ps) c", ps=self.patch_size)
```

把压缩的隐向量还原回 token 级别的表示。如果 `patch_size=1`，这步是恒等变换。

### 3.3 KV Cache 支持

解码器支持 per-sample KV cache（`modeling_cola_vae.py:331-334`）：

```python
self._k_cache: Optional[list[torch.Tensor]] = None
self._v_cache: Optional[list[torch.Tensor]] = None
```

推理时，前缀的 K/V 被缓存，后续 block 只需要计算新 block 的 Q。

---

## 四、RoPE 位置编码

### 4.1 VAE 的 RoPE 实现

代码位置：`modeling_cola_vae.py:146-221`

VAE 使用自己实现的 RoPE（`VAERotaryEmbedding`），不依赖外部库：

```python
# modeling_cola_vae.py:174
inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, ...) / dim))
```

关键参数：`rope_theta=500000`（`configuration_cola_vae.py:88`）。

### 4.2 变长序列的位置索引

代码位置：`modeling_cola_vae.py:126-138`

```python
def _build_na_positions(txt_shape):
    """每个 sample 的 positions 从 0 重新开始"""
    parts = [torch.arange(int(l), device=txt_shape.device)
             for l in txt_shape.flatten()]
    return torch.cat(parts).unsqueeze(0)

def _build_na_q_positions(txt_shape, txt_q_shape):
    """Q 的位置对齐到 K 的尾部"""
    parts = []
    for k_len, q_len in zip(...):
        parts.append(torch.arange(k_len - q_len, k_len, ...))
    return torch.cat(parts).unsqueeze(0)
```

这是 NA（no-padding）布局的核心：每个样本的位置从 0 开始，Q 的位置对齐到 K 的尾部。

---

## 五、Stage 1 训练目标

论文式 2.2.1 给出 Stage 1 的损失：

$$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z_0|x)}[\log p_\theta(x|z_0)] + \beta \cdot \text{KL}(q_\phi(z_0|x) \| p_{\text{base}}(z_0)) + \lambda_{\text{mask}} \cdot \mathcal{L}_{\text{mask}}$$

三项各自的含义：

| 项 | 作用 | 如果去掉会怎样 |
|----|------|--------------|
| 重构损失 | 让解码器能从隐向量还原 token | 解码器学不到有意义的映射 |
| KL 正则 | 让隐空间接近先验分布 $p_{\text{base}}$ | 隐空间不规整，无法做扩散 |
| BERT mask 损失 | 防止编码器"偷懒"（只编码表面信息） | 编码器可能只保留局部信息，丢失全局语义 |

**BERT mask 损失的作用**：如果只有重构损失，编码器可能学到一个"查表"式的映射——每个 token 对应一个固定的隐向量，不需要理解上下文。BERT mask 损失强制编码器在部分 token 被 mask 的情况下也能编码出有意义的隐向量，从而迫使它学习上下文理解。

---

## 六、隐空间缩放

代码位置：`inference.py:408`

```python
latents_list = [((lat - shift) * scale).float() for lat in enc.latents_list]
```

其中 `shift = vae.shifting_factor`，`scale = vae.scaling_factor`（默认 0.0 和 1.0）。

这两个常数的作用是调整隐空间的几何形状，使得 DiT 先验更容易拟合。类似于图像扩散中对 latent 做归一化。

---

## 七、配置参数对照

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `vocab_size` | 100278 | 词表大小 |
| `encoder_num_blocks` | 4 | encoder Transformer 层数 |
| `decoder_num_blocks` | 4 | decoder Transformer 层数 |
| `dim` | 1536 | 模型维度 |
| `ffn_dim` | 6144 | FFN 中间层维度 |
| `latent_dim` | 16 | 隐空间维度 |
| `patch_size` | 1 | patchify 因子 |
| `num_heads` | 12 | 注意力头数 |
| `rope_theta` | 500000 | RoPE 频率基数 |
| `block_size` | 4 | 分块大小 |
| `block_causal` | True | 是否使用分块因果注意力 |

---

## 八、面试追问清单

**基础（⭐）**：

1. VAE 的 ELBO 损失由哪几项组成？
2. Conv1d patchify 在 Cola Text VAE 中的作用是什么？
3. 为什么推理时用 `mode()` 而不是 `sample()`？

**进阶（⭐⭐）**：

4. BERT mask 损失为什么能防止编码器"偷懒"？
5. VAE 的 `rope_theta=500000` 和 DiT 的 `theta=10000` 有什么区别？
6. `latent_dim=16` 是否太小？增大对生成质量有什么影响？

**专家（⭐⭐⭐）**：

7. VAE 的 KL 项用 `p_base` 而不是标准的 $\mathcal{N}(0, I)$，这对隐空间有什么影响？
8. 如果去掉 VAE，直接在 token embedding 空间做扩散，会有什么问题？
9. Stage 2 的 reference-encoder KL 正则是如何防止隐空间漂移的？

---

## 九、下期预告

下一章我们将深入 DiT 先验——看看它是如何在隐空间里做 Flow Matching 的，分块因果注意力的具体实现，以及 CFG（Classifier-Free Guidance）的工作原理。

---

> **系列导航**
>
> [第 01 章：语言生成的三次范式之争](01_generation_paradigm.md) · [第 02 章：扩散模型 10 分钟速通](02_diffusion_foundation.md) · [第 03 章：离散扩散的困境](03_discrete_diffusion.md) · [第 04 章：Cola DLM 架构全景](04_cola_architecture.md)
>
> **第 05 章：Text VAE 深度解剖** ← 你在这里
>
> [第 06 章：分块因果 DiT 先验](06_dit_prior.md) · [第 07 章：推理流水线逐行拆解](07_inference_pipeline.md) · [第 08 章：工程实现评析](08_engineering.md) · [第 09 章：评测复现与结果深度分析](09_benchmark.md) · [第 10 章：从文本到多模态](10_future.md)

---

> **作者**：[Yunzenn](https://github.com/Yunzenn) 
