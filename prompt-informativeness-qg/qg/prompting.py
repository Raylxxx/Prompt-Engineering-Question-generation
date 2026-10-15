"""由 prompt、shots 和当前答案拼接模型输入，并解析模型输出。"""

import re

from . import config

SYSTEM_MESSAGE = "You are an assistant that writes the question a given text is answering."


def build_messages(prompt: str, shots: list[dict], answer: str) -> list[dict]:
    """构造一次生成调用的对话消息。

    prompt : 20 个候选 prompt 之一（A..T）
    shots  : 示例列表，每项为 {"question", "answer"}，可以为空
    answer : 需要为其生成问题的答案文本
    """
    parts = [prompt]
    if shots:
        parts.append("Here are some examples of a text and the original question it answers:")
        for i, s in enumerate(shots, 1):
            parts.append(f"Example {i}\nText: {s['answer']}\nQuestion: {s['question']}")
    parts.append(
        f"Now apply the instruction to the following text. Output exactly {config.N_QUESTIONS} "
        f"questions, one per line, numbered 1 to {config.N_QUESTIONS}, with nothing else.\n"
        f"Text: {answer}"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*(.+?)\s*$")


def parse_questions(text: str, n: int = config.N_QUESTIONS) -> list[str]:
    """从编号或项目符号列表中提取最多 n 个问题。"""
    questions = []
    for line in text.splitlines():
        m = _NUMBERED.match(line)
        if m:
            questions.append(m.group(1))
    if not questions:  # 没有编号时，退而取所有非空行
        questions = [line.strip() for line in text.splitlines() if line.strip()]
    return questions[:n]
