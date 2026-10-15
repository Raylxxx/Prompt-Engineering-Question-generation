"""绘制 shot 数量与生成质量的关系曲线。

默认读取 results/paper/table9_shot_sweep.csv 中的数据作图；
加 --from-outputs 时改用本地实验结果（outputs/summary/<metric>_by_dataset.csv）。

用法：
    python -m qg.plot_shots
    python -m qg.plot_shots --from-outputs --prompt M --method max_entropy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config

LABELS = {"truthfulqa": "TruthfulQA", "alpaca": "Alpaca-cleaned"}


def paper_curves() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(config.ROOT / "results" / "paper" / "table9_shot_sweep.csv")
    return {
        "cosine": df.melt(id_vars="shots", value_vars=["cosine_truthfulqa", "cosine_alpaca"],
                          var_name="dataset", value_name="score")
                    .assign(dataset=lambda d: d["dataset"].str.replace("cosine_", "")),
        "gpt4": df.melt(id_vars="shots", value_vars=["gpt4_truthfulqa", "gpt4_alpaca"],
                        var_name="dataset", value_name="score")
                  .assign(dataset=lambda d: d["dataset"].str.replace("gpt4_", "")),
    }


def output_curves(prompt: str, method: str) -> dict[str, pd.DataFrame]:
    curves = {}
    for metric in ("cosine", "gpt4"):
        path = config.OUTPUT_DIR / "summary" / f"{metric}_by_dataset.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[(df["prompt_id"] == prompt) & (df["method"] == method)]
            curves[metric] = df.rename(columns={"k": "shots"})[["shots", "dataset", "score"]]
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-outputs", action="store_true")
    parser.add_argument("--prompt", default="M")
    parser.add_argument("--method", default="max_entropy")
    parser.add_argument("--out", default=str(config.ROOT / "results" / "figures" / "shot_sweep.png"))
    args = parser.parse_args()

    curves = output_curves(args.prompt, args.method) if args.from_outputs else paper_curves()
    titles = {"cosine": "Max cosine similarity", "gpt4": "GPT-4 score"}

    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4))
    axes = [axes] if len(curves) == 1 else axes
    for ax, (metric, df) in zip(axes, curves.items()):
        for dataset, d in df.groupby("dataset"):
            d = d.sort_values("shots")
            ax.plot(d["shots"], d["score"], marker="o", label=LABELS.get(dataset, dataset))
        ax.set_title(titles[metric])
        ax.set_xlabel("Number of shots")
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Prompt-{args.prompt} + {args.method.replace('_', ' ')} shots")
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
