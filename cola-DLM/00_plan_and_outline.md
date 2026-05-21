# Cola DLM 博客系列 — 计划与大纲

> **系列定位**：以 Cola DLM（arXiv:2605.06548）为核心案例，梳理语言生成范式从 AR → 离散扩散 → 连续隐空间扩散的演进，深入解剖"在连续隐空间做扩散"这一技术路线的设计哲学、工程实现与未来前景。
>
> **目标读者**：有一定深度学习基础、了解 Transformer 基本原理、对扩散模型感兴趣的研究者与工程师。
>
> **写作风格**：参考 `transformer/` 系列的学术精读风格，加入 `agent/` 系列的对比评测视角。问题先行、公式配数值例子、每章对比表 + Mermaid 架构图 + 面试追问。

---

## 系列目录

| 章节 | 文件名 | 核心困惑 | 状态 |
|------|--------|---------|------|
| 01 | `01_generation_paradigm.md` | AR 已经这么强了，为什么还要搞扩散语言模型？ | 待写 |
| 02 | `02_diffusion_foundation.md` | Cola DLM 用的 Flow Matching 和经典 DDPM 有什么关系？ | 待写 |
| 03 | `03_discrete_diffusion.md` | LLaDA/MDLM 已经证明离散扩散能工作，为什么还要"绕路"去连续空间？ | 待写 |
| 04 | `04_cola_architecture.md` | 为什么要分成 VAE + DiT + Decoder 三个组件？ | 待写 |
| 05 | `05_text_vae.md` | Text VAE 怎么把离散 token 变成连续向量？隐空间长什么样？ | 待写 |
| 06 | `06_dit_prior.md` | DiT 怎么学习隐空间先验？分块因果注意力怎么实现？ | 待写 |
| 07 | `07_inference_pipeline.md` | `generate_task_repaint_inference` 这 453 行代码到底在做什么？ | 待写 |
| 08 | `08_engineering.md` | 哪些设计值得学习，哪些需要改进？ | 待写 |
| 09 | `09_benchmark.md` | 26.75% 的 Task Average 到底意味着什么？ | 待写 |
| 10 | `10_future.md` | 扩散语言模型的下一步是什么？ | 待写 |

---

## 核心类比（贯穿全系列）

将 Cola DLM 比作一个**跨语言写作工作室**：

- **VAE Encoder（翻译官 q_φ）**：把人类语言（离散 token）翻译成"世界语"（连续隐向量）
- **DiT 先验（世界语作家 p_ψ）**：在世界语空间里创作——先写大纲（第一个 block），再逐段填充（后续 block），每段都能看到前面所有段落
- **VAE Decoder（翻译官的逆 p_θ）**：把世界语作品翻译回人类语言

---

## 技术校验记录

以下结论已对照 `cola-DLMxiangmu/` 源码逐行验证：

| 技术点 | 源码位置 | 验证结果 |
|--------|---------|---------|
| 分块因果 mask：block 内双向、block 间因果 | `attention_utils.py:149-150` | `q_block >= k_block` + `same_sample` |
| Euler 更新 `z_{t-Δ} = z_t - Δ/T · v_ψ` | `inference.py:357,649` | `dt = (t_curr - t_next) / T`，`txt = txt - drift * dt` |
| CFG 融合 `pred = uncond + scale × (cond - uncond)` | `inference.py:648` | `drift = s * (drift_cond - drift_uncond) + drift_uncond` |
| 短 prompt CFG 退化 `guidance_scale → 1.0` | `inference.py:546-554` | 空 prefix 时 per-sample scale = 1.0 |
| VAE 用 LayerNorm（非 RMSNorm） | `modeling_cola_vae.py:239-242` | `build_norm_layer("layer_norm", ...)` |
| DiT 用 GELU tanh（非 SwiGLU） | `modeling_cola_dit.py:349` | `nn.GELU("tanh")` |
| VAE rope_theta=500000，DiT theta=10000 | `configuration_cola_vae.py:88` / `modeling_cola_dit.py:252` | 两个模型位置编码频率特性不同 |
| `generate_task_repaint_inference` 453 行 | `inference.py:285-738` | 单一函数处理全流程 |
| 服务端全局锁串行 | `server.py:132,162` | `threading.Lock` |
| 无 Flash Attention | `modeling_cola_dit.py:390` | `query.mul(scale) @ key.transpose(-2, -1)` |
| 8 个 benchmark 准确率 | `eval_output/accuracy_summary.csv` | 全部匹配 |
