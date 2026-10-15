"""把逐条样本的分数汇总成结果表。

对每种指标（cosine、gpt4）输出：
    outputs/summary/<metric>_by_dataset.csv  按 数据集 × prompt × 方法 × shot 数 求平均
    outputs/summary/<metric>_table.csv       行为 prompt、列为（方法, shot 数），对两个数据集取平均
并打印得分最高的 5 个组合。

用法：
    python -m qg.summarize
"""

import pandas as pd

from . import config
from .utils import iter_jsonl

METHOD_ORDER = ["most_helpful", "max_entropy", "mixed", "random", "none"]


def load_scores(metric: str) -> pd.DataFrame:
    rows = []
    for dataset in config.DATASETS:
        rows.extend(iter_jsonl(config.OUTPUT_DIR / "scores" / f"{metric}_{dataset}.jsonl"))
    return pd.DataFrame(rows)


def summarize(metric: str) -> None:
    df = load_scores(metric)
    if df.empty:
        print(f"[{metric}] no scores found")
        return
    out_dir = config.OUTPUT_DIR / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_dataset = (df.groupby(["dataset", "prompt_id", "method", "k"])["score"]
                    .mean().reset_index())
    by_dataset.to_csv(out_dir / f"{metric}_by_dataset.csv", index=False)

    # 两个数据集的平均分再取平均
    overall = by_dataset.groupby(["prompt_id", "method", "k"])["score"].mean().reset_index()
    table = overall.pivot_table(index="prompt_id", columns=["method", "k"], values="score")
    table = table.reindex(columns=sorted(table.columns, key=lambda c: (METHOD_ORDER.index(c[0]), c[1])))
    table.to_csv(out_dir / f"{metric}_table.csv", float_format="%.4f")

    print(f"\n=== {metric}: prompts x (method, shots), mean over datasets ===")
    print(table.round(4).to_string())
    print(f"\n=== {metric}: top 5 configurations ===")
    print(overall.sort_values("score", ascending=False).head(5).round(4).to_string(index=False))
    print(f"\n=== {metric}: best prompt per (method, shots) ===")
    print(overall.groupby(["method", "k"])["score"].max().round(4).to_string())


def main() -> None:
    for metric in ("cosine", "gpt4"):
        summarize(metric)


if __name__ == "__main__":
    main()
