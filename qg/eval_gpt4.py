"""评估 2：GPT-4 打分。

GPT-4 按 1~5 分的评分标准，判断每个生成问题与参考问题的接近程度。
与余弦相似度一样，每条输入取 5 个问题中的最高分。

输出：outputs/scores/gpt4_<dataset>.jsonl（可断点续跑）

用法：
    python -m qg.eval_gpt4
    python -m qg.eval_gpt4 --methods max_entropy --prompts M   # 只给 shot 数量实验打分
"""

import argparse
import json
import re

from tqdm import tqdm

from . import config
from .llm import chat
from .utils import append_jsonl, iter_jsonl

RUBRIC = """\
1 - The meaning is completely different from the preset question; there may be serious misunderstandings. The information expressed by the two is completely inconsistent.
2 - It overlaps somewhat with the preset question, but the core idea is different. There is some resonance in certain aspects, but the main theme and purpose are clearly different.
3 - Close to the meaning of the preset question, with slight differences in some details. The overall idea is similar, but word choice or structure differs slightly.
4 - Highly similar to the preset question, with only small differences in wording. The core ideas are almost identical; the changes in vocabulary or expression do not affect understanding.
5 - The meaning is exactly the same as the preset question, with almost no substantial difference. Both convey the same information and views."""

JUDGE_TEMPLATE = """\
You are evaluating automatically generated questions.
Compare each generated question with the preset question and give it a score from 1 to 5 using this rubric:

{rubric}

Preset question:
{reference}

Generated questions:
{questions}

Return only a JSON object of the form {{"scores": [s1, s2, ...]}} with one integer per generated question, in order."""


def judge(reference: str, questions: list[str], model: str) -> list[int]:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    prompt = JUDGE_TEMPLATE.format(rubric=RUBRIC, reference=reference, questions=numbered)
    raw = chat([{"role": "user", "content": prompt}], model, temperature=0.0, max_tokens=100)
    try:
        scores = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["scores"]
    except (AttributeError, ValueError, KeyError, TypeError):
        scores = re.findall(r"[1-5]", raw)
    return [min(5, max(1, int(s))) for s in scores][: len(questions)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS), choices=config.DATASETS)
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None, choices=config.SHOT_METHODS)
    parser.add_argument("--model", default=config.JUDGE_MODEL)
    args = parser.parse_args()

    for dataset in args.datasets:
        gens = list(iter_jsonl(config.OUTPUT_DIR / "generations" / f"{dataset}.jsonl"))
        gens = [g for g in gens
                if (args.prompts is None or g["prompt_id"] in args.prompts)
                and (args.methods is None or g["method"] in args.methods)]
        out_path = config.OUTPUT_DIR / "scores" / f"gpt4_{dataset}.jsonl"
        done = {(r["sample_id"], r["prompt_id"], r["method"], r["k"]) for r in iter_jsonl(out_path)}
        todo = [g for g in gens if (g["sample_id"], g["prompt_id"], g["method"], g["k"]) not in done]
        print(f"{dataset}: {len(todo)} generations to score ({len(done)} already done)")

        for g in tqdm(todo, desc=dataset):
            scores = judge(g["reference"], g["questions"], args.model) if g["questions"] else []
            append_jsonl(out_path, {
                "dataset": dataset,
                "sample_id": g["sample_id"],
                "prompt_id": g["prompt_id"],
                "method": g["method"],
                "k": g["k"],
                "scores": scores,
                "score": max(scores) if scores else 1,
            })


if __name__ == "__main__":
    main()
