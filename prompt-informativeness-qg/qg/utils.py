"""文件读写工具函数。"""

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompts(path: Path) -> dict[str, str]:
    """读取 prompt 文件，返回 {prompt 编号: prompt 文本}（A..T）。"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {p["id"]: p["text"] for p in items}
