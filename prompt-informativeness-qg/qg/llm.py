"""OpenAI 对话接口的简单封装，带失败重试。"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY
    return _client


def chat(messages: list[dict], model: str, temperature: float = 0.0,
         max_tokens: int = 512, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # 限流、超时、服务端临时错误等
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  OpenAI error ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    return ""
