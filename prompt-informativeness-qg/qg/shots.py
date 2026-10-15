"""Demonstration（shot）构造方法。

所有选择函数的输入：
    query_vec : (d,)   当前输入答案的归一化向量
    pool_vecs : (n, d) 候选池中各答案的归一化向量
    k         : shot 数量
返回候选池中被选中的 k 个下标。

相似度只在*答案*之间计算，因为推理时模型能看到的只有答案。
"""

import numpy as np

from . import config


def most_helpful(query_vec: np.ndarray, pool_vecs: np.ndarray, k: int, **_) -> list[int]:
    """(1) Most helpful shots：选与输入答案最相似的 k 个答案。

    相似的答案往往对应相似的问题，这些 shot 能把目标问题的模式提前“透露”给模型。
    """
    sims = pool_vecs @ query_vec
    return np.argsort(-sims)[:k].tolist()


def max_entropy(query_vec: np.ndarray, pool_vecs: np.ndarray, k: int, **_) -> list[int]:
    """(2) Maximum entropy shots：贪心地选出最不相似的答案。

    第 1 个 shot 是与输入答案最不相似的答案，之后每个 shot 都是与上一个已选 shot
    最不相似的答案。目标是让 shot 集合覆盖尽可能多的情况、互不冗余。
    """
    selected: list[int] = []
    available = np.ones(len(pool_vecs), dtype=bool)
    anchor = query_vec
    for _ in range(min(k, len(pool_vecs))):
        sims = pool_vecs @ anchor
        sims[~available] = np.inf
        idx = int(np.argmin(sims))
        selected.append(idx)
        available[idx] = False
        anchor = pool_vecs[idx]
    return selected


def mixed(query_vec: np.ndarray, pool_vecs: np.ndarray, k: int,
          n_candidates: int = config.MIXED_CANDIDATES, **_) -> list[int]:
    """(3) Mixed method：先保证相关，再保证多样。

    1. 取与输入答案最相似的 `n_candidates` 个答案作为候选。
    2. 在候选中先选与输入最不相似的一个。
    3. 之后反复选出与“输入 + 已选 shots”综合相似度最低的候选。
    """
    candidates = np.argsort(-(pool_vecs @ query_vec))[:n_candidates]
    cand_vecs = pool_vecs[candidates]
    score = cand_vecs @ query_vec          # 累计的综合相似度
    available = np.ones(len(candidates), dtype=bool)
    selected: list[int] = []
    for _ in range(min(k, len(candidates))):
        s = np.where(available, score, np.inf)
        j = int(np.argmin(s))
        selected.append(int(candidates[j]))
        available[j] = False
        score = score + cand_vecs @ cand_vecs[j]
    return selected


def random_shots(query_vec: np.ndarray, pool_vecs: np.ndarray, k: int,
                 rng: np.random.Generator | None = None, **_) -> list[int]:
    """对照组：从候选池中随机抽取 k 个 shot。"""
    rng = rng or np.random.default_rng(config.SEED)
    return rng.choice(len(pool_vecs), size=min(k, len(pool_vecs)), replace=False).tolist()


def no_shots(*_, **__) -> list[int]:
    """对照组：不加 shot（zero-shot）。"""
    return []


SELECTORS = {
    "most_helpful": most_helpful,
    "max_entropy": max_entropy,
    "mixed": mixed,
    "random": random_shots,
    "none": no_shots,
}


def select_shots(method: str, query_vec: np.ndarray, pool_vecs: np.ndarray, k: int, **kwargs) -> list[int]:
    if method not in SELECTORS:
        raise ValueError(f"Unknown shot method {method!r}; choose from {list(SELECTORS)}")
    if method == "none" or k == 0:
        return []
    return SELECTORS[method](query_vec, pool_vecs, k, **kwargs)
