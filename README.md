# 面向条件式问题生成的 Prompt 信息量优化

[中文](README.md) | [English](README.en.md)

**一句话总结**：问题生成（Question Generation）任务中，大模型拿到的是一段*答案*，需要还原出*原始问题*。当 prompt 已经把任务说清楚以后，再改写 prompt 的措辞收效甚微。本项目换一个角度：通过挑选加入 prompt 的 **few-shot 示例（demonstration）** 来提升生成问题的质量。项目在 2 个数据集、20 个 prompt 上，对比了三种 shot 选择策略：最相似、最多样，以及二者结合。

## 亮点

- **3 种 demonstration 构造方法**：*Most Helpful Shots*（相似性）、*Maximum Entropy Shots*（多样性）、*Mixed*（先相关、再多样），并与 *Random* shots、*No shots* 两组对照比较。
- **20 个 prompt 的候选集**：4 个来自已有工作的问题生成 prompt，加上由大模型改写出的 16 个。
- **两种评估方式**：与参考问题的最大余弦相似度，以及 GPT-4 按 1~5 分评分标准打分。
- **主要发现**
  - 在 5 shots 下，*Maximum Entropy* 在 20 个 prompt 上的平均余弦相似度最高。
  - 最佳配置的余弦相似度比朴素 baseline prompt（0.8003）高约 **+0.03**。
  - **shot 数量存在最佳区间，大约在 5~7 个**，继续增加反而变差。
  - 要求生成 "complex / intricate / sophisticated"（复杂）问题的 prompt 表现一直较差，因为这类措辞会把模型带离原问题。

## 方法

```mermaid
flowchart LR
    A["4 个基础 QG prompt<br/>(A–D)"] --> B["大模型改写<br/>→ 20 个 prompt (A–T)"]
    B --> C["选择 shots<br/>most helpful / max entropy /<br/>mixed / random / none"]
    C --> D["gpt-3.5-turbo<br/>生成 5 个问题"]
    D --> E["评估<br/>最大余弦相似度<br/>+ GPT-4 打分 (1–5)"]
```

### 1. Prompt 候选集（[`prompts/prompts.json`](prompts/prompts.json)）

起点是 [Amyeen, 2023](https://arxiv.org/abs/2310.18867) 提出的 4 个问题生成 prompt：

| 编号 | Prompt |
|---|---|
| A | Generate 5 questions from the given text. *（baseline：最简单，模拟普通用户）* |
| B | Generate 5 complex questions derived from the text. |
| C | Generate 5 questions from the text; make sure the questions can be answered. |
| D | Generate 5 questions from the text; answer the question in the text; if the question is answered in the context, output 5 questions. |

每个基础 prompt 由大模型以 4 种方式改写（*Adaptation、Variation、Modification、Twist*），意思保持不变，只改变措辞和句式，得到 E~T（[`qg/expand_prompts.py`](qg/expand_prompts.py)）。每个 prompt 都要求生成 5 个问题，其中最匹配的一个计入得分。

### 2. Demonstration（shot）构造（[`qg/shots.py`](qg/shots.py)）

每个候选 shot 是一个（问题，答案）对。答案用 spaCy 词向量表示，彼此之间用余弦相似度比较。

| 方法 | 如何选出 k 个 shot | 直觉 |
|---|---|---|
| **Most Helpful** | 选与输入答案**最相似**的 k 个答案 | 相似的答案会把目标问题的模式“透露”给模型 |
| **Maximum Entropy** | 贪心：先选与输入最不相似的答案，再选与上一个 shot 最不相似的答案，依此类推 | 尽量覆盖更多情况，且不重复 |
| **Mixed** | 先取与输入最相似的 20 个答案作为候选，再在候选中贪心地选出与“输入 + 已选 shots”综合相似度最低的样本 | 既相关又多样 |
| Random | 随机选 k 个答案 | 对照组 |
| No shots | 零样本 | 对照组 |

### 3. 评估

- **余弦相似度**（[`qg/eval_cosine.py`](qg/eval_cosine.py)）：对每条输入，计算参考问题与 5 个生成问题的最大相似度；先在所有输入上取平均，再对两个数据集取平均。
- **GPT-4 打分**（[`qg/eval_gpt4.py`](qg/eval_gpt4.py)）：GPT-4 按 1~5 分的评分标准，把每个生成问题与参考问题对比打分。1 分表示“意思完全不同”，5 分表示“意思完全相同”。每条输入取 5 个分数中的最高分。

### 4. 数据（[`qg/prepare_data.py`](qg/prepare_data.py)）

| 数据集 | 规模 | 特点 | 清洗方式 |
|---|---|---|---|
| [TruthfulQA](https://huggingface.co/datasets/truthful_qa) | 817 | 问题长、答案短 | 只用 `generation` 部分；`multiple_choice` 的答案包含多个选项，故剔除 |
| [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | 51,760 | 问题短、答案长 | 剔除 `input` 非空的数据，因为仅凭答案无法还原原问题 |

每个数据集随机抽取 50 个问答对作为模型输入，并逐条人工检查，确认答案能追溯回对应问题。

## 实验结果（论文报告的数值）

原始表格见 [`results/paper/`](results/paper/)。

### 余弦相似度：20 个 prompt 的平均

| 方法 | 2 shots | 5 shots |
|---|---|---|
| Most Helpful | 0.7977 | 0.8109 |
| **Maximum Entropy** | 0.7932 | **0.8170** |
| Mixed | 0.7871 | 0.7904 |
| Random | 0.8052 | 0.7851 |
| No shots | 0.7914 | – |

Baseline（Prompt-A，无 shots）：**0.8003**。

### 余弦相似度：最佳配置

| 排名 | 配置 | 余弦相似度 |
|---|---|---|
| 1 | Prompt-P + Max Entropy（5 shots）| 0.8334 |
| 2 | Prompt-D + Most Helpful（5 shots）| 0.8315 |
| 3 | Prompt-F + Max Entropy（5 shots）| 0.8312 |
| 4 | Prompt-M + Max Entropy（5 shots）| 0.8306 |
| 5 | Prompt-N + Max Entropy（5 shots）| 0.8303 |

### GPT-4 打分：各方法下最佳 prompt 的得分

| 方法 | 2 shots | 5 shots |
|---|---|---|
| Most Helpful | 4.38 | 4.49 |
| **Maximum Entropy** | 4.375 | **4.50** |
| Mixed | 4.465 | 4.44 |
| Random | 4.42 | 4.41 |
| No shots | 4.365 | – |

### shot 数量的影响（Prompt-M + Maximum Entropy）

| Shots | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 余弦 – TruthfulQA | 0.7941 | 0.8093 | 0.8245 | 0.8284 | 0.8235 | 0.8246 | **0.8305** | 0.8235 | 0.7917 | 0.8022 |
| 余弦 – Alpaca-cleaned | 0.7722 | 0.8040 | 0.7901 | 0.7858 | 0.8112 | **0.8314** | 0.8081 | 0.8120 | 0.8176 | 0.7922 |
| GPT-4 – TruthfulQA | 4.34 | 4.29 | 4.36 | 4.41 | **4.46** | 4.44 | 4.39 | 4.34 | 4.36 | 4.29 |
| GPT-4 – Alpaca-cleaned | 4.28 | 4.29 | 4.22 | 4.34 | **4.51** | 4.39 | 4.17 | 4.43 | 4.40 | 4.34 |

质量先上升，在约 5~7 个 shot 时达到峰值，之后下降。运行 `python -m qg.plot_shots` 可画出这些曲线。

## 仓库结构

`qg` 是 **Q**uestion **G**eneration 的缩写，是本项目的 Python 包名。

```
.
├── paper/                      # 论文 PDF
├── prompts/prompts.json        # 20 个候选 prompt（A–T）
├── qg/
│   ├── config.py               # 路径、模型名、默认参数
│   ├── prepare_data.py         # 下载、清洗、抽样数据集
│   ├── expand_prompts.py       # 用大模型把 A–D 改写成 E–T
│   ├── embeddings.py           # spaCy 向量 + 余弦相似度
│   ├── shots.py                # most helpful / max entropy / mixed / random 选 shot
│   ├── prompting.py            # 拼接模型输入、解析生成的问题
│   ├── llm.py                  # 带重试的 OpenAI 调用封装
│   ├── generate.py             # 运行问题生成（可断点续跑）
│   ├── eval_cosine.py          # 评估 1：最大余弦相似度
│   ├── eval_gpt4.py            # 评估 2：GPT-4 按评分标准打分
│   ├── summarize.py            # 把分数汇总成论文格式的表
│   └── plot_shots.py           # shot 数量曲线
├── results/paper/              # 论文报告的数值（CSV）
├── requirements.txt
└── .env.example
```

## 快速开始

需要 Python 3.10+。

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cp .env.example .env            # 然后在 .env 中填入 OpenAI key，或设置环境变量 OPENAI_API_KEY
```

```bash
# 1. 数据：生成 data/{truthfulqa,alpaca}_{eval,pool}.jsonl
python -m qg.prepare_data

# 2. 主实验：20 个 prompt × 4 种方法 × {2, 5} shots，外加 no shots
python -m qg.generate --methods most_helpful max_entropy mixed random --shots 2 5
python -m qg.generate --methods none

# 3. shot 数量消融：Prompt-M + max entropy，1~10 个 shot
python -m qg.generate --prompts M --methods max_entropy --shots 1 2 3 4 5 6 7 8 9 10

# 4. 评估与汇总
python -m qg.eval_cosine
python -m qg.eval_gpt4
python -m qg.summarize
python -m qg.plot_shots --from-outputs
```

生成和 GPT-4 打分的结果会追加写入 `outputs/`，并跳过已完成的部分，中断后可以接着跑。

> **费用提示**：完整实验为 2 个数据集 × 50 条输入 × 20 个 prompt × 9 种配置 = 18,000 次生成调用，GPT-4 打分还需要同样数量的调用。建议先用 `--prompts` 和 `--methods` 跑一个子集。

## 局限性

- 每个数据集只有 50 条输入，每种配置只运行了一次。很多方法之间的差距很小（余弦相似度 ≤ 0.03），应视为趋势，而不是统计上显著的差异。
- 生成模型只用了 `gpt-3.5-turbo`，向量是 spaCy 词向量的平均。
- “Maximum entropy” 是一种贪心的多样性启发式方法，并没有真正计算熵。
- GPT-4 的打分没有与人工评判做过校准。

## 引用

```bibtex
@misc{xie2024prompt,
  title  = {Prompt Informativeness Optimization for Conditional Question Generation},
  author = {Xie, Diyang},
  year   = {2024},
  note   = {China Jiliang University}
}
```

## 参考文献

- R. Amyeen. *Prompt-Engineering and Transformer-based Question Generation and Evaluation.* arXiv:2310.18867, 2023.
- S. Lin, J. Hilton, O. Evans. *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* arXiv:2109.07958, 2022.
- R. Taori et al. *Stanford Alpaca: An Instruction-following LLaMA model.* 2023.
- J. White et al. *A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT.* arXiv:2302.11382, 2023.

## 许可证

[MIT](LICENSE)（[中文译本](LICENSE.zh-CN.md)）
