"""对每种（数据集, prompt, shot 方法, shot 数量）组合生成问题。

每条评估答案与一个 prompt、以及从候选池中选出的 k 个 shot 拼成输入，交给生成模型，
模型返回 5 个问题。结果追加写入 outputs/generations/<dataset>.jsonl，
已完成的组合会自动跳过，中断后可以接着运行。

用法：
    # 主实验：20 个 prompt × 4 种方法 × {2, 5} 个 shot，外加无 shot
    python -m qg.generate --methods most_helpful max_entropy mixed random --shots 2 5
    python -m qg.generate --methods none

    # shot 数量实验：Prompt-M + max entropy，1~10 个 shot
    python -m qg.generate --prompts M --methods max_entropy --shots 1 2 3 4 5 6 7 8 9 10
"""

import argparse

import numpy as np
from tqdm import tqdm

from . import config
from .embeddings import embed_cached
from .llm import chat
from .prompting import build_messages, parse_questions
from .shots import select_shots
from .utils import append_jsonl, iter_jsonl, load_prompts, read_jsonl


def run_key(sample_id: str, prompt_id: str, method: str, k: int) -> tuple:
    return (sample_id, prompt_id, method, 0 if method == "none" else k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS), choices=config.DATASETS)
    parser.add_argument("--prompts", nargs="+", default=None, help="prompt 编号（默认全部 A..T）")
    parser.add_argument("--methods", nargs="+", default=list(config.SHOT_METHODS), choices=config.SHOT_METHODS)
    parser.add_argument("--shots", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--model", default=config.GENERATOR_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    prompts = load_prompts(config.PROMPTS_FILE)
    prompt_ids = args.prompts or list(prompts)

    for dataset in args.datasets:
        eval_rows = read_jsonl(config.DATA_DIR / f"{dataset}_eval.jsonl")
        pool_rows = read_jsonl(config.DATA_DIR / f"{dataset}_pool.jsonl")
        eval_vecs = embed_cached([r["answer"] for r in eval_rows], f"{dataset}_eval")
        pool_vecs = embed_cached([r["answer"] for r in pool_rows], f"{dataset}_pool")

        out_path = config.OUTPUT_DIR / "generations" / f"{dataset}.jsonl"
        done = {run_key(r["sample_id"], r["prompt_id"], r["method"], r["k"]) for r in iter_jsonl(out_path)}

        jobs = []
        for method in args.methods:
            for k in ([0] if method == "none" else args.shots):
                for pid in prompt_ids:
                    for i, row in enumerate(eval_rows):
                        if run_key(row["id"], pid, method, k) not in done:
                            jobs.append((method, k, pid, i))

        print(f"{dataset}: {len(jobs)} generations to run ({len(done)} already done)")
        for method, k, pid, i in tqdm(jobs, desc=dataset):
            row = eval_rows[i]
            rng = np.random.default_rng([args.seed, i, k])
            idx = select_shots(method, eval_vecs[i], pool_vecs, k, rng=rng)
            shots = [pool_rows[j] for j in idx]
            messages = build_messages(prompts[pid], shots, row["answer"])
            raw = chat(messages, args.model, temperature=args.temperature)
            append_jsonl(out_path, {
                "dataset": dataset,
                "sample_id": row["id"],
                "prompt_id": pid,
                "method": method,
                "k": k,
                "shot_ids": [s["id"] for s in shots],
                "reference": row["question"],
                "answer": row["answer"],
                "questions": parse_questions(raw),
                "raw": raw,
                "model": args.model,
            })


if __name__ == "__main__":
    main()
