# 第 07 章：推理流水线逐行拆解 —— 从 prompt 到生成文本

> **论文**：[*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)
> **项目地址**：[ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM)
> **源码**：[`inference.py`](https://github.com/ByteDance-Seed/Cola-DLM/blob/main/cola_dlm/inference.py)
>
> **核心困惑**：`generate_task_repaint_inference` 这 454 行代码到底在做什么？

---

## 一、宏观流程

推理严格对应论文式 2.2.4–2.2.6 的三步流程：

```
输入 prompt: "The capital of France is"
        │
        ▼
  ┌─────────────┐
  │ Step 1: 分词  │  tokenizer.encode(prompt)
  │ + block 对齐  │  pad 到 patch_size*block_size 的倍数
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Step 2: 前缀  │  z_pre = vae.encode(input_ids)
  │ 编码         │  + latent label 分类
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────┐
  │ Step 3: 分块先验传输          │  for block in blocks:
  │ (DiT + CFG + Euler ODE)     │    noise → DiT → z_0
  └──────┬──────────────────────┘
         │
         ▼
  ┌─────────────┐
  │ Step 4: 条件  │  logits = vae.decode(z_0)
  │ 解码 + 采样   │  tokens = sample(logits)
  └──────┬──────┘
         │
         ▼
  输出: "Paris"
```

---

## 二、Step 1：分词与 block 对齐

代码位置：`inference.py:367-397`

```python
chunk = patch_size * block_size  # 1 * 4 = 4
for item in prompts:
    ids = tokenizer.encode(prompt_str).ids

    # 对齐到 chunk 的整数倍
    p_pad_len = (chunk - len(ids) % chunk) % chunk
    t_labels = [1] * len(ids) + [3] * p_pad_len  # 1=prompt, 3=pad
    ids = ids + [pad_token_id] * p_pad_len
```

**latent label 分类**：
- `1` = prompt token（需要被编码）
- `2` = 待生成 token
- `3` = 硬 pad（对齐用，后续被裁掉）

---

## 三、Step 2：前缀编码

代码位置：`inference.py:404-408`

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    enc = vae.encode(input_ids_list)
    latents_list = [((lat - shift) * scale).float() for lat in enc.latents_list]
```

- `vae.encode()` 返回 `TextVAEEncoderOutput`，包含 `latents_list`（每样本的隐向量）
- 隐向量做缩放 `(z - shift) * scale`，转为 fp32

---

## 四、Step 3：分块先验传输（核心循环）

### 4.1 噪声初始化

代码位置：`inference.py:578-606`

```python
# 默认：全局随机噪声
txt = torch.randn(batch_size * block_size, latent_dim, device=device)

# 可选：per-sample 确定性噪声（通过环境变量启用）
if per_sample_seed_env:
    for b, item in enumerate(prompts):
        g = torch.Generator(device=device)
        g.manual_seed(base_seed + sid_int * 1_000 + step * 10_000_000)
        noise_3d[b] = torch.randn(block_size, latent_dim, device=device, generator=g)
```

### 4.2 Euler ODE 求解

代码位置：`inference.py:608-654`

```python
timesteps = torch.linspace(int(T), 0, timestep_num + 1)  # [1000, 937.5, ..., 0]

for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
    ts_batch = torch.full((txt.shape[0],), t_curr, device=device)
    dt = (float(t_curr) - float(t_next)) / max(T, 1.0)

    # 条件前向（用 KV cache）
    drift_cond = dit(txt=txt_bf16, txt_shape=txt_shape_cum,
                     txt_q_shape=txt_q_shape, timestep=ts_bf16,
                     update_kv=False, use_kv_cache=True).txt_sample

    # 无条件前向（不用 cache）
    drift_uncond = dit(txt=txt_bf16, txt_shape=txt_q_shape,
                       txt_q_shape=txt_q_shape, timestep=ts_bf16,
                       update_kv=False, use_kv_cache=False).txt_sample

    # CFG 融合
    s = cfg_scale_first_block if step == 0 else guidance_scale
    drift = s * (drift_cond - drift_uncond) + drift_uncond

    # Euler 更新: z_{t-Δ} = z_t - (Δ/T) * v_ψ
    txt_next = txt - drift * dt
    txt = txt_next
```

**关键细节**：每个 block 需要做 16 步 × 2 次（cond + uncond）= 32 次 DiT 前向。

### 4.3 KV Cache 提交

代码位置：`inference.py:690-700`

```python
# 把刚去噪的 block 提交到 DiT KV cache
txt_bf16 = txt.to(torch.bfloat16)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _ = dit(txt=txt_bf16, txt_shape=txt_shape_cum,
            txt_q_shape=txt_q_shape,
            timestep=torch.zeros(...),
            update_kv=True,   # ← 关键：提交到 cache
            use_kv_cache=True)
```

---

## 五、Step 4：条件解码 + 采样

代码位置：`inference.py:656-675`

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    decoded = vae.decode(z=txt, txt_shape=txt_shape_cum,
                         txt_q_shape=txt_q_shape, update_kv=True)

decoded_logits = decoded.view(batch_size, block_size * patch_size, -1)
one_block_ids = sample_with_strategies(
    decoded_logits, generated_ids=context_ids,
    temperature=temperature, top_k=top_k, top_p=top_p,
    repetition_penalty=repetition_penalty,
)
```

### 5.1 采样策略

代码位置：`inference.py:211-266`

```python
def sample_with_strategies(logits, generated_ids=None, temperature=0.8,
                           top_k=50, top_p=0.95, repetition_penalty=1.1):
    # 1. 重复惩罚
    if repetition_penalty != 1.0:
        score = torch.gather(logits, 1, target_gen_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(1, target_gen_ids, score)

    # 2. 温度缩放
    if temperature != 1.0:
        logits = logits / temperature

    # 3. Top-k 过滤
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < min_values, float("-inf"), logits)

    # 4. Top-p (nucleus) 过滤
    if 0 < top_p < 1.0:
        # 按概率累积截断
        cumulative_probs = torch.softmax(sorted_logits).cumsum()
        ...

    # 5. 采样
    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
```

---

## 六、Prompt 模板

代码位置：`inference.py:80-203`

每个 benchmark 任务有特定的 few-shot 模板：

```python
def apply_prompt_template(task, context, question, answer, choices):
    if task == "lambada":
        return question  # 直接用原文
    elif task == "squad":
        return f"Context: ...\nQuestion: {question}\nAnswer:"
    elif task == "mmlu":
        return f"Question: {question}\n{choices_text}\nAnswer:"
    # ... 每个任务有不同的模板
```

**注意**：模板对生成质量影响很大。论文中的准确率是在特定模板下测得的。

---

## 七、循环终止条件

代码位置：`inference.py:682-706`

```python
# EOS 检测
for b in range(one_block_ids.shape[0]):
    if eos_token_id is not None and eos_token_id in one_block_ids[b]:
        eos_status[b] = True
if eos_status.all():
    stop_flag = True

# max_new_tokens 检测
if step * block_size * patch_size >= max_new_tokens:
    stop_flag = True
```

---

## 八、完整流程图

```
prompt: "The capital of France is"
    │
    ▼
分词: [464, 3139, 286, 4881, 318]  (5 tokens)
    │
    ▼
block 对齐: [464, 3139, 286, 4881, 318, 100277, 100277, 100277]  (→ 8 tokens, 2 blocks)
labels:     [1,    1,    1,   1,    1,   3,       3,       3    ]
    │
    ▼
VAE encode: z_pre shape = (8, 16)  → 裁掉 label-3 尾部 → (5, 16)
    │
    ▼
latent label: [1, 1, 1, 1, 1]  (5 个 prompt latent)
prefix:       z_pre[:4]  (前 4 个，block 对齐)
first block:  z_pre[4:] + random pad  (第 5 个 + 3 个噪声)
    │
    ▼
DiT prefix prefetch: 把 prefix 写入 KV cache
    │
    ▼
Block 1 生成:
  noise ~ N(0, I)  →  16 步 Euler ODE  →  z_0^(1)  →  VAE decode  →  logits  →  "Paris"
  提交 z_0^(1) 到 KV cache
    │
    ▼
Block 2 生成（如果需要更多 token）:
  noise ~ N(0, I)  →  16 步 Euler ODE  →  z_0^(2)  →  VAE decode  →  logits  →  ...
    │
    ▼
输出: "Paris"
```

---

## 九、面试追问清单

**基础（⭐）**：

1. 推理的三步流程分别对应论文的哪些公式？
2. latent label 的 1/2/3 分别代表什么？
3. 为什么需要 block 对齐？

**进阶（⭐⭐）**：

4. CFG 的条件前向和无条件前向有什么区别？
5. per-sample noise seed 的作用是什么？
6. 为什么第一个 block 的 CFG scale 可能被设为 1.0？

**专家（⭐⭐⭐）**：

7. 每个 block 的 16 步 Euler ODE 求解能否减少步数？trade-off 是什么？
8. KV cache 的 update_kv 和 use_kv_cache 的组合逻辑是什么？
9. 如果 prompt 长度正好是 block_size 的整数倍，推理流程有什么简化？

---

## 十、下期预告

下一章我们将从工程角度评析 Cola DLM 的实现——哪些设计值得学习，哪些需要改进，以及服务化部署的短板。

---

> **系列导航**
>
> [第 01 章](01_generation_paradigm.md) · [第 02 章](02_diffusion_foundation.md) · [第 03 章](03_discrete_diffusion.md) · [第 04 章](04_cola_architecture.md) · [第 05 章](05_text_vae.md) · [第 06 章](06_dit_prior.md)
>
> **第 07 章：推理流水线逐行拆解** ← 你在这里
>
> [第 08 章](08_engineering.md) · [第 09 章](09_benchmark.md) · [第 10 章](10_future.md)

---

> **作者**：[Yunzenn](https://github.com/Yunzenn) 
