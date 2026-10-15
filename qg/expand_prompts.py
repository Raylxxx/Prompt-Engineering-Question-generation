"""自动生成 prompt 候选集。

以 A~D 四个问题生成 prompt 为模板，让大模型对每个模板按四种方式改写
（Adaptation、Variation、Modification、Twist）：保持语义不变，只改变措辞和句式。
改写结果与 A~D 合起来构成 20 个候选 prompt。

实验所用的 20 个 prompt 已保存在 prompts/prompts.json，运行本脚本会生成一套新的候选集。

用法：
    python -m qg.expand_prompts --out prompts/prompts_regenerated.json
"""

import argparse
import json
from pathlib import Path

from . import config
from .llm import chat
from .utils import load_prompts

STYLES = ("Adaptation", "Variation", "Modification", "Twist")

META_PROMPT = (
    "Here is a prompt used to make a language model generate questions from a text:\n"
    '"{prompt}"\n\n'
    "Rewrite it as a new prompt ({style}). Keep the meaning and the requirement of "
    "generating 5 questions roughly unchanged, but change the wording and the syntactic "
    "structure. Output only the new prompt."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.GENERATOR_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=config.ROOT / "prompts" / "prompts_regenerated.json")
    args = parser.parse_args()

    base = {pid: text for pid, text in load_prompts(config.PROMPTS_FILE).items() if pid in "ABCD"}
    items = [{"id": pid, "source": "base", "text": text} for pid, text in base.items()]
    next_id = ord("E")
    for pid, text in base.items():
        for style in STYLES:
            messages = [{"role": "user", "content": META_PROMPT.format(prompt=text, style=style)}]
            new_text = chat(messages, args.model, temperature=args.temperature).strip().strip('"')
            items.append({"id": chr(next_id), "source": f"{pid} {style}", "text": new_text})
            print(f"{chr(next_id)} ({pid} {style}): {new_text}")
            next_id += 1

    args.out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(items)} prompts to {args.out}")


if __name__ == "__main__":
    main()
