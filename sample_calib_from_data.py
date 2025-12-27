import json
import os
import random
import re
from typing import Iterable, List, Optional, Tuple


def _iter_jsonl(path: str) -> Iterable[dict]:
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 容错：部分 jsonl 可能存在意外前缀/尾随内容（例如行首出现杂字符）。
            # 找到第一个 '{' 作为 JSON 起点，然后用 raw_decode 解析首个 JSON 对象。
            start = line.find("{")
            if start == -1:
                continue
            try:
                obj, _end = decoder.raw_decode(line[start:])
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _extract_user_text(obj: dict) -> Optional[str]:
    """从一条训练样本中提取用户侧文本。

    兼容常见两类格式：
    - {"messages": [{"role":"user","content":"..."}, ...]}
    - {"prompt": "..."} / {"instruction": "..."} / {"question": "..."} / {"input": "..."} / {"text": "..."}
    """

    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if (m.get("role") or "").strip().lower() == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()

    for k in ("prompt", "instruction", "question", "input", "text"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def _normalize_question(q: str) -> str:
    s = (q or "").strip()
    s = re.sub(r"\s+", " ", s)

    # 去掉一些常见前缀提示（数据里重复很多）
    prefixes = [
        "简答：",
        "只给结论：",
        "要点式回答：",
        "为了评测请用最短答案回答：",
        "请直接输出关键术语+结论：",
        "请用最短答案回答：",
        "请直接给结论：",
    ]
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p) :].strip()
            break

    return s


def sample_calib(
    data_path: str,
    n: int = 512,
    seed: int = 42,
    min_len: int = 6,
    max_len: int = 512,
) -> Tuple[List[str], int]:
    """从 data.jsonl 里抽取校准文本。

    - 以 user 文本为主
    - 先做归一化去重（避免同一题不同“简答/只给结论”占满样本）
    - 最终随机采样 n 条

    返回：texts, total_candidates
    """

    rng = random.Random(seed)

    seen = set()
    candidates: List[str] = []

    for obj in _iter_jsonl(data_path):
        q = _extract_user_text(obj)
        if not q:
            continue

        qn = _normalize_question(q)
        if len(qn) < int(min_len):
            continue

        # 过长文本截断（按字符；量化工具通常会再做 tokenizer 截断）
        if len(qn) > int(max_len):
            qn = qn[: int(max_len)].rstrip()

        if qn in seen:
            continue
        seen.add(qn)
        candidates.append(qn)

    total = len(candidates)
    if total <= n:
        texts = candidates
    else:
        texts = rng.sample(candidates, n)

    return texts, total


def main() -> int:
    data_path = os.environ.get("DATA_JSONL", "data.jsonl")
    out_jsonl = os.environ.get("OUT_JSONL", "calib_512.jsonl")
    out_txt = os.environ.get("OUT_TXT", "calib_512.txt")
    n = int(os.environ.get("N", "512"))
    seed = int(os.environ.get("SEED", "42"))

    texts, total = sample_calib(data_path=data_path, n=n, seed=seed)
    print(f"[sample_calib] data={data_path}, unique_candidates={total}, sampled={len(texts)}")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")

    print("[sample_calib] wrote:", out_jsonl)
    print("[sample_calib] wrote:", out_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
