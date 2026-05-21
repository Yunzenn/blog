# 第 09 章：评测复现与结果深度分析

> **论文**：[*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)
> **项目地址**：[ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM)
> **源码**：[`scripts/run_benchmark.sh`](https://github.com/ByteDance-Seed/Cola-DLM/blob/main/scripts/run_benchmark.sh)
>
> **核心困惑**：26.75% 的 Task Average 到底意味着什么？Scaling 曲线说明了什么？

---

## 一、评测设置

### 1.1 模型规模

| 组件 | 参数量 | 配置 |
|------|--------|------|
| Text VAE | ~500M | 4 encoder + 4 decoder blocks, dim=1536, 12 heads |
| DiT 先验 | ~1.8B | 24 layers, 16 heads, head_dim=128, txt_dim=2048 |
| **总计** | **~2.3B** | |

训练计算量：约 2000 EFLOPs（论文 RQ4 scaling 曲线中最大的 checkpoint）。

### 1.2 评测任务

| 任务 | 测什么 | 评测方式 |
|------|--------|---------|
| LAMBADA | 词汇预测（最后一个词） | Exact match |
| MMLU | 多领域知识 | Choice letter extraction |
| OBQA | 常识科学问答 | Choice letter extraction |
| HellaSwag | 常识推理（句子补全） | Choice letter extraction |
| RACE | 阅读理解 | Choice letter extraction |
| SIQA | 社会交互推理 | Choice letter extraction |
| SQuAD | 阅读理解（抽取式） | Similarity match |
| Story Cloze | 故事结尾选择 | Choice letter extraction |

### 1.3 推理参数

| 参数 | 值 |
|------|-----|
| `max_new_tokens` | 32 |
| `timestep_num` | 16 |
| `guidance_scale` | 7.0 |
| `temperature` | 0.0（greedy） |
| `top_k` | 50 |
| `top_p` | 0.9 |

---

## 二、结果分析

### 2.1 准确率总览

来源：`eval_output/accuracy_summary.csv`

| 任务 | 准确率（%） | 分析 |
|------|-------------|------|
| LAMBADA | **50.80** | 词汇预测，依赖局部上下文，表现最好 |
| SQuAD | **30.90** | 抽取式问答，需要定位能力 |
| Story Cloze | **30.77** | 故事推理，需要全局理解 |
| SIQA | **28.90** | 社会推理，需要常识 |
| OBQA | **23.00** | 科学常识，需要知识 |
| RACE | **19.60** | 阅读理解，较长上下文 |
| MMLU | **19.30** | 多领域知识，接近随机（25%） |
| HellaSwag | **10.70** | 常识推理，表现最差 |
| **平均** | **26.75** | |

### 2.2 逐任务深度分析

**LAMBADA（50.80%）— 表现最好的任务**

LAMBADA 的任务是预测句子的最后一个词。这是一个**纯局部上下文**任务——模型只需要看前面的词就能预测最后一个词。Cola DLM 的 VAE encoder 能很好地编码局部上下文，所以表现最好。

**MMLU（19.30%）— 接近随机**

MMLU 是多领域知识问答（4 选 1，随机 = 25%）。19.30% 甚至低于随机，说明：
- 2B 参数的模型知识量有限
- Cola DLM 的预训练目标（Flow Matching + VAE 重构）不是为知识存储设计的
- 没有经过指令微调，不擅长选择题格式

**HellaSwag（10.70%）— 表现最差的任务**

HellaSwag 是常识推理（4 选 1，随机 = 25%）。10.70% 远低于随机，说明：
- 常识推理需要细粒度的局部建模
- 扩散模型的"全局规划"优势在这个任务上没有体现
- 可能是 prompt 模板不适配

### 2.3 Scaling 曲线解读

论文图 2 显示了 Cola DLM vs AR vs LLaDA 的 scaling 曲线：

**关键观察**：
1. Cola DLM 的 Task Average 曲线**仍在上升**，而 AR 趋于饱和
2. 在偏推理任务（MMLU、RACE、Story Cloze、OBQA）上，Cola DLM 有明显优势
3. 在偏常识任务（HellaSwag）上，Cola DLM 表现较差

**解读**：
- 扩散模型的"全局规划"能力在推理任务上更有优势
- 但在需要大量世界知识的任务上，2B 参数的 Cola DLM 不如 AR
- Cola DLM 的 scaling 潜力可能比 AR 更大（曲线仍在上升）

---

## 三、复现步骤

### 3.1 环境准备

```bash
# 安装依赖
pip install -e .

# 下载模型
# hf_models/cola_dlm/cola_dit/
# hf_models/cola_dlm/cola_vae/
# hf_models/tokenizer.json

# 下载评测数据
# generate_task_data/*.jsonl
```

### 3.2 运行评测

```bash
# 全部 8 个任务
bash scripts/run_benchmark.sh

# 单任务
TASKS="lambada" NUM_GPUS=1 bash scripts/run_benchmark.sh
```

### 3.3 计算准确率

```bash
python scripts/acc_calc.py
```

---

## 四、局限性诚实讨论

### 4.1 绝对性能低

26.75% 的 Task Average 与同规模 AR 模型差距明显。即使是 LAMBADA 的 50.80%，也远低于 GPT-2（77%+）等模型。

### 4.2 仅短输出

默认 `max_new_tokens=32`，只能生成短回答。长文本生成的质量未验证。

### 4.3 Prompt 格式敏感

每个任务需要特定的 few-shot 模板。模板变化可能导致准确率大幅波动。

### 4.4 训练数据和计算量

2000 EFLOPs 的训练量相对较小。更大的训练量可能显著提升性能（scaling 曲线仍在上升）。

---

## 五、面试追问清单

**基础（⭐）**：

1. Cola DLM 在哪些任务上表现好？为什么？
2. Scaling 曲线"仍在上升"意味着什么？
3. 为什么 MMLU 准确率接近随机？

**进阶（⭐⭐）**：

4. Cola DLM 和 LLaDA 的 scaling 曲线差异说明了什么？
5. 如何解释 HellaSwag 的低准确率？
6. `max_new_tokens=32` 对评测结果有什么影响？

**专家（⭐⭐⭐）**：

7. 如果训练量从 2000 EFLOPs 增加到 20000 EFLOPs，Cola DLM 的性能会如何变化？
8. Cola DLM 的评测协议和标准 AR 模型的评测协议有什么区别？
9. 如何设计一个更公平的扩散语言模型 vs AR 模型对比实验？

---

## 六、下期预告

最后一章，我们将展望扩散语言模型的未来——多模态统一、scale-up 的可能性、以及与 RLHF 的结合。

---

> **系列导航**
>
> [第 01 章](01_generation_paradigm.md) · [第 02 章](02_diffusion_foundation.md) · [第 03 章](03_discrete_diffusion.md) · [第 04 章](04_cola_architecture.md) · [第 05 章](05_text_vae.md) · [第 06 章](06_dit_prior.md) · [第 07 章](07_inference_pipeline.md) · [第 08 章](08_engineering.md)
>
> **第 09 章：评测复现与结果深度分析** ← 你在这里
>
> [第 10 章：从文本到多模态](10_future.md)

---

> **作者**：[Yunzenn](https://github.com/Yunzenn) 
