"""评估 1：余弦相似度。

对每条生成结果，用 spaCy 计算参考问题与 5 个生成问题的向量，取余弦相似度的*最大值*。
各组合的平均分由 `qg.summarize` 统一计算。

输出：outputs/scores/cosine_<dataset>.jsonl

用法：
    python -m qg.eval_cosine
"""

import argparse

import numpy as np

from . import config
from .embeddings import embed
from .utils import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS), choices=config.DATASETS)
    args = parser.parse_args()

    for dataset in args.datasets:
        gens = list(iter_jsonl(config.OUTPUT_DIR / "generations" / f"{dataset}.jsonl"))
        if not gens:
            print(f"{dataset}: no generations found, skipping")
            continue

        # 相同文本只计算一次向量
        texts = sorted({g["reference"] for g in gens} | {q for g in gens for q in g["questions"]})
        vecs = dict(zip(texts, embed(texts)))

        rows = []
        for g in gens:
            ref = vecs[g["reference"]]
            sims = [float(vecs[q] @ ref) for q in g["questions"]]
            rows.append({
                "dataset": dataset,
                "sample_id": g["sample_id"],
                "prompt_id": g["prompt_id"],
                "method": g["method"],
                "k": g["k"],
                "sims": sims,
                "score": max(sims) if sims else 0.0,
            })
        write_jsonl(config.OUTPUT_DIR / "scores" / f"cosine_{dataset}.jsonl", rows)
        print(f"{dataset}: scored {len(rows)} generations, mean max-cosine = "
              f"{np.mean([r['score'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
