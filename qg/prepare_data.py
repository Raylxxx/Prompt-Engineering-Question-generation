"""下载、清洗并抽样两个数据集。

每个数据集输出两个文件：
    data/<name>_pool.jsonl   清洗后的问答对，作为 shot 候选池
    data/<name>_eval.jsonl   抽取的 N_EVAL_SAMPLES 条问答对，作为模型输入

清洗规则：
  * TruthfulQA：只使用 `generation` 部分。`multiple_choice` 部分的答案包含多个
    选项，模型无法判断哪一个是输入，因此不用。
  * Alpaca-cleaned：剔除 `input` 字段非空的数据。额外的 input 属于原问题的一部分，
    仅凭答案无法还原出原问题。

抽样后需人工检查每条数据能否由答案追溯回原问题；不能追溯的，用 `--exclude-ids`
剔除后重新抽样。

用法：
    python -m qg.prepare_data
    python -m qg.prepare_data --exclude-ids alpaca-123 truthfulqa-45
"""

import argparse
import random

from datasets import load_dataset

from . import config
from .utils import write_jsonl


def load_truthfulqa() -> list[dict]:
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rows = []
    for i, r in enumerate(ds):
        answer = (r["best_answer"] or "").strip()
        question = (r["question"] or "").strip()
        if answer and question:
            rows.append({"id": f"truthfulqa-{i}", "question": question, "answer": answer})
    return rows


def load_alpaca() -> list[dict]:
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = []
    for i, r in enumerate(ds):
        if (r["input"] or "").strip():
            continue  # 带额外 input 的数据无法由答案还原问题
        question = (r["instruction"] or "").strip()
        answer = (r["output"] or "").strip()
        if answer and question:
            rows.append({"id": f"alpaca-{i}", "question": question, "answer": answer})
    return rows


LOADERS = {"truthfulqa": load_truthfulqa, "alpaca": load_alpaca}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=config.N_EVAL_SAMPLES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--exclude-ids", nargs="*", default=[],
                        help="人工检查不通过的样本 ID，剔除后重新抽样")
    args = parser.parse_args()
    excluded = set(args.exclude_ids)

    for name, loader in LOADERS.items():
        rows = [r for r in loader() if r["id"] not in excluded]
        rng = random.Random(args.seed)
        eval_rows = rng.sample(rows, args.n)
        eval_ids = {r["id"] for r in eval_rows}
        # 从候选池中去掉评估样本，避免样本被选为自己的 shot
        pool_rows = [r for r in rows if r["id"] not in eval_ids]

        write_jsonl(config.DATA_DIR / f"{name}_eval.jsonl", eval_rows)
        write_jsonl(config.DATA_DIR / f"{name}_pool.jsonl", pool_rows)
        print(f"{name}: {len(rows)} cleaned rows -> {len(eval_rows)} eval, {len(pool_rows)} pool")


if __name__ == "__main__":
    main()
