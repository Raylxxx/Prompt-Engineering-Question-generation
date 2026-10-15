"""用 spaCy 计算文本向量，并缓存到磁盘。"""

import hashlib
from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _nlp(model: str = config.SPACY_MODEL):
    import spacy

    # 只需要分词和词向量，关闭其余组件以加快速度
    nlp = spacy.load(model)
    nlp.select_pipes(disable=nlp.pipe_names)
    return nlp


def embed(texts: list[str], model: str = config.SPACY_MODEL, batch_size: int = 256) -> np.ndarray:
    """返回 (n, d) 的文本向量矩阵，每行已做 L2 归一化，点积即余弦相似度。"""
    nlp = _nlp(model)
    vecs = np.stack([doc.vector for doc in nlp.pipe(texts, batch_size=batch_size)]).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_cached(texts: list[str], tag: str, model: str = config.SPACY_MODEL) -> np.ndarray:
    """计算 `texts` 的向量，按 tag + 内容哈希缓存在 .cache/ 下。"""
    digest = hashlib.md5(("\n".join(texts) + model).encode("utf-8")).hexdigest()[:12]
    path = config.CACHE_DIR / f"emb_{tag}_{digest}.npy"
    if path.exists():
        return np.load(path)
    vecs = embed(texts, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs
