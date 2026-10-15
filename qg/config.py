"""全局路径与默认参数。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPTS_FILE = ROOT / "prompts" / "prompts.json"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / ".cache"

DATASETS = ("truthfulqa", "alpaca")

# 生成模型与打分模型，可通过命令行参数 --model 覆盖
GENERATOR_MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4"

# 计算答案向量所用的 spaCy 模型（需带词向量：md 或 lg）
SPACY_MODEL = "en_core_web_lg"

# 实验默认参数
N_EVAL_SAMPLES = 50          # 每个数据集抽取的问答对数量
SEED = 42
N_QUESTIONS = 5              # 每个 prompt 要求模型生成的问题数
MIXED_CANDIDATES = 20        # mixed 方法的候选池大小

SHOT_METHODS = ("most_helpful", "max_entropy", "mixed", "random", "none")
