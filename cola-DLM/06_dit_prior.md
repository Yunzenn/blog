# 第 06 章：分块因果 DiT 先验 —— 在隐空间里做 Flow Matching

> **论文**：[*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)
> **项目地址**：[ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM)
> **源码**：[`modeling_cola_dit.py`](https://github.com/ByteDance-Seed/Cola-DLM/blob/main/cola_dlm/modeling_cola_dit.py)
>
> **核心困惑**：DiT 是怎么学习隐空间先验 $p_\psi(z_0)$ 的？分块因果注意力怎么实现？CFG 怎么工作？

---

## 一、DiT 在图像领域的成功

DiT（Diffusion Transformer）由 Scalable Diffusion Models with Transformers（Peebles & Xie, 2023）提出，用 Transformer 替代 UNet 作为扩散模型的骨干网络。核心思想：

- 把图像切成 patch → 线性投影 → Transformer blocks → 线性投影 → unpatch
- 用 AdaLN（Adaptive Layer Norm）注入时间步信息

Cola DLM 把这个思路从 2D 图像迁移到 1D 文本隐序列。

---

## 二、模型架构

### 2.1 整体结构

代码位置：`modeling_cola_dit.py:536-690`

```
输入: txt (L_q_total, in_channels)
        │
        ▼
┌──────────────┐
│  PatchIn1D   │  patchify + 线性投影
└──────┬───────┘
       │ (L_q_total/patch_size, txt_dim)
       ▼
┌──────────────┐
│ TimestepEmb  │  sinusoidal → MLP → emb_dim
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ DiTBlock ×24 │  AdaLN + Attention + FFN
│ (分块因果)    │  per-sample KV cache
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ PatchOut1D   │  线性投影 + unpatch
└──────┬───────┘
       │
       ▼
输出: txt_sample (L_q_total, out_channels)
```

### 2.2 配置参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `txt_in_channels` | 16 | 输入隐空间维度（= VAE 的 `latent_dim`） |
| `txt_out_channels` | 16 | 输出隐空间维度 |
| `txt_dim` | 2048 | Transformer 隐藏维度 |
| `emb_dim` | 2048 | AdaLN 条件维度 |
| `heads` | 16 | 注意力头数 |
| `head_dim` | 128 | 每头维度 |
| `expand_ratio` | 4 | FFN 扩展比 |
| `num_layers` | 24 | Transformer 层数 |
| `patch_size` | 1 | patchify 因子 |
| `rope_dim` | 96 | RoPE 作用的通道数（< head_dim=128） |
| `block_size` | 4 | 分块大小 |

总参数量：约 1.8B（24 层 × 16 头 × 128 head_dim = 2048 hidden dim）。

---

## 三、关键组件

### 3.1 PatchIn1D / PatchOut1D

代码位置：`modeling_cola_dit.py:166-205`

```python
class PatchIn1D(nn.Module):
    def __init__(self, in_channels, patch_size, dim):
        self.proj = nn.Linear(in_channels * patch_size, dim)

    def forward(self, txt, txt_shape):
        txt_shape_before_patchify = txt_shape
        if self.patch_size != 1:  # patch_size=1 时跳过，直接投影
            batch_list = _unflatten(txt, txt_shape)
            for i in range(len(batch_list)):
                batch_list[i] = rearrange(batch_list[i], "(T t) c -> T (t c)", t=self.patch_size)
            txt, txt_shape = _flatten(batch_list)
        txt = self.proj(txt)
        return txt, txt_shape, txt_shape_before_patchify
```

默认 `patch_size=1`，所以 rearrange 被完全跳过，只有线性投影生效。

### 3.2 TimestepEmbedding（AdaLN 条件化）

代码位置：`modeling_cola_dit.py:135-158`

```python
class TimestepEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim, hidden_dim, output_dim):
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, timestep, device, dtype):
        emb = _get_sinusoidal_embedding(timestep, self.sinusoidal_dim)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        emb = self.proj_out(emb)
        return emb
```

时间步 $t$ 通过 sinusoidal embedding + MLP 注入到每个 Transformer block 的 AdaLN 中。

### 3.3 AdaLN（Adaptive Layer Norm）

代码位置：`modeling_cola_dit.py:299-336`

```python
class AdaLN(nn.Module):
    def forward(self, hid, emb, layer, mode, norm_layer=None, residual=None, **kwargs):
        emb = getattr(self, f"{layer}_{mode}")(emb)  # 线性投影
        if mode == "in":
            shift, scale = emb.chunk(2, dim=-1)
            return norm_layer(hid) * (1 + scale) + shift  # scale + shift
        if mode == "out":
            return hid * emb + residual  # gate + residual
```

每个 block 有两处 AdaLN：
- `mode="in"`：在 attention/FFN 之前，做 scale + shift
- `mode="out"`：在 attention/FFN 之后，做 gate + residual

### 3.4 MLP

代码位置：`modeling_cola_dit.py:344-352`

```python
class MLP(nn.Module):
    def __init__(self, dim, expand_ratio):
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")  # 注意：不是 SwiGLU！
        self.proj_out = nn.Linear(dim * expand_ratio, dim)
```

**注意**：DiT 用 **GELU tanh**，而 VAE 用 **SwiGLU**。这是一个设计选择差异。

---

## 四、分块因果注意力

### 4.1 ColaDiTAttention

代码位置：`modeling_cola_dit.py:360-463`

```python
class ColaDiTAttention(nn.Module):
    def __init__(self, txt_dim, heads, head_dim, qk_bias, qk_norm_eps, rope_dim):
        self.proj_qkv = nn.Linear(txt_dim, inner_dim * 3)
        self.norm_q = nn.LayerNorm(head_dim)  # QK-norm
        self.norm_k = nn.LayerNorm(head_dim)
        self.rope = TextRotaryEmbedding(dim=rope_dim)  # rope_dim=96 < head_dim=128
```

**RoPE 只作用于部分通道**：`rope_dim=96`，而 `head_dim=128`。这意味着 128 个通道中有 96 个有位置编码，32 个没有。

### 4.2 KV Cache 管理

代码位置：`modeling_cola_dit.py:420-442`

```python
# per-sample KV cache
self._k_cache: Optional[list[torch.Tensor]] = None
self._v_cache: Optional[list[torch.Tensor]] = None

# forward 中的逻辑：
if update_kv:  # 提交新 block 到 cache
    self._k_cache = [torch.cat([c, n], dim=0) for c, n in zip(self._k_cache, new_ks)]
    full_k = torch.cat(self._k_cache, dim=0)
elif use_kv_cache and self._k_cache is not None:  # 读 cache
    full_k = torch.cat([torch.cat([c, n], dim=0) for c, n in zip(self._k_cache, new_ks)], dim=0)
else:  # 无 cache
    full_k = txt_k
```

### 4.3 注意力计算

代码位置：`modeling_cola_dit.py:381-397`

```python
def slow_attn(self, query, key, value, attn_mask=None):
    d_head = query.shape[-1]
    device_type = "cuda" if query.is_cuda else query.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        scale = 1.0 / (d_head ** 0.5)
        attn = query.mul(scale) @ key.transpose(-2, -1)  # 显式构造完整矩阵
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn.dtype)
        attn_weight = attn.softmax(dim=-1)
        attn_out = attn_weight @ value
    return attn_out
```

**注意**：没有使用 Flash Attention，而是显式构造完整的 $(L_q, L_k)$ 注意力矩阵。这是当前实现的一个重要限制。

---

## 五、CFG（Classifier-Free Guidance）

### 5.1 原理

CFG 是扩散模型的标准技巧：同时做条件生成和无条件生成，然后按比例混合。

$$\hat{v} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

其中 $s$ 是 guidance scale（默认 7.0）。

### 5.2 代码实现

代码位置：`inference.py:621-648`

```python
# 条件前向：用 KV cache（能看到 prompt 和历史 block）
drift_cond = dit(txt=txt_bf16, txt_shape=txt_shape_cum,
                 txt_q_shape=txt_q_shape, timestep=ts_bf16,
                 update_kv=False, use_kv_cache=True).txt_sample

# 无条件前向：不用 cache（只能看到当前 block）
drift_uncond = dit(txt=txt_bf16, txt_shape=txt_q_shape,
                   txt_q_shape=txt_q_shape, timestep=ts_bf16,
                   update_kv=False, use_kv_cache=False).txt_sample

# CFG 融合
s = cfg_scale_first_block if step == 0 else guidance_scale
drift = s * (drift_cond - drift_uncond) + drift_uncond
```

### 5.3 短 prompt 的 CFG 退化

代码位置：`inference.py:531-554`

当 prompt 短于 `block_size` 时，第一个生成 block 的前缀 KV cache 为空，条件和无条件前向数学上相同。此时 CFG 会放大 bf16 噪声：

```python
cfg_scale_first_block = torch.tensor(
    [guidance_scale if pl > 0 else 1.0 for pl in prefix_lens],
    device=device, dtype=torch.bfloat16,
).repeat_interleave(block_size).unsqueeze(-1)
```

空 prefix 的样本自动将 `guidance_scale` 降为 1.0。

---

## 六、ColaDiTBlock

代码位置：`modeling_cola_dit.py:471-519`

每个 block 的前向流程：

```python
def forward(self, txt, *, txt_shape, txt_q_shape, emb, ...):
    # 1. AdaLN + Attention
    txt_msa = self.ada(txt, emb=emb, layer="msa", mode="in", norm_layer=self.msa_norm)
    txt_msa = self.msa(txt_msa, txt_shape=txt_shape, txt_q_shape=txt_q_shape, ...)
    txt = self.ada(txt_msa, emb=emb, layer="msa", mode="out", residual=txt)

    # 2. AdaLN + FFN
    txt_mlp = self.ada(txt, emb=emb, layer="mlp", mode="in", norm_layer=self.mlp_norm)
    txt_mlp = self.mlp(txt_mlp)
    txt = self.ada(txt_mlp, emb=emb, layer="mlp", mode="out", residual=txt)
    return txt
```

---

## 七、Stage 2 联合训练目标

论文式 2.2.3 给出 Stage 2 的损失：

$$\mathcal{L}_{\text{stage2}} = \lambda_{\text{VAE}} \cdot \mathcal{L}_{\text{VAAE}} + \lambda_{\text{FM}} \cdot \mathcal{L}_{\text{FM}} + \lambda_{\text{ref}} \cdot \mathbb{E}[\text{KL}(q_\phi(z_0|x) \| q_{\phi_{\text{ref}}}(z_0|x))]$$

| 项 | 作用 |
|----|------|
| $\mathcal{L}_{\text{VAE}}$ | 保持 VAE 的重构能力 |
| $\mathcal{L}_{\text{FM}}$ | 训练 DiT 先验（Flow Matching） |
| reference KL | 防止 VAE 的隐空间漂移（对齐冻结的参考编码器） |

---

## 八、面试追问清单

**基础（⭐）**：

1. DiT 和 UNet 作为扩散模型骨干的区别是什么？
2. AdaLN 是如何注入时间步信息的？
3. CFG 的 guidance scale 对生成质量有什么影响？

**进阶（⭐⭐）**：

4. 为什么 DiT 的 RoPE 只作用于 96/128 个通道？
5. per-sample KV cache 和标准 KV cache 有什么区别？
6. Stage 2 的 reference-encoder KL 正则为什么能防止隐空间漂移？

**专家（⭐⭐⭐）**：

7. DiT 用 GELU tanh 而 VAE 用 SwiGLU，这个差异会影响什么？
8. `rope_theta=10000`（DiT）vs `rope_theta=500000`（VAE）的位置编码频率差异意味着什么？
9. 如果把 block_size 从 4 改为 16，DiT 的注意力模式会怎么变化？

---

## 九、下期预告

下一章我们将逐行拆解推理流水线——从 prompt 输入到文本输出的完整过程，包括 tokenization、前缀编码、分块先验传输、条件解码和采样策略。

---

> **系列导航**
>
> [第 01 章](01_generation_paradigm.md) · [第 02 章](02_diffusion_foundation.md) · [第 03 章](03_discrete_diffusion.md) · [第 04 章](04_cola_architecture.md) · [第 05 章](05_text_vae.md)
>
> **第 06 章：分块因果 DiT 先验** ← 你在这里
>
> [第 07 章](07_inference_pipeline.md) · [第 08 章](08_engineering.md) · [第 09 章](09_benchmark.md) · [第 10 章](10_future.md)

---

> **作者**：[Yunzenn](https://github.com/Yunzenn) 
