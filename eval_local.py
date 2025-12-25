import argparse
import json
import os
import re
import time
from typing import List, Tuple, Dict, Any, Optional

import requests
from docx import Document
import jieba
from rouge_score import rouge_scorer
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def parse_docx_qa(docx_path: str) -> List[Tuple[str, str]]:
    """
    Parse docx where Q/A are presented as:
      问题：...
      答案：...
    Ignore other headings/paragraphs.
    """
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]

    qa: List[Tuple[str, str]] = []
    cur_q: Optional[str] = None
    cur_a: Optional[str] = None

    q_pat = re.compile(r"^\s*问题[:：]\s*(.*)\s*$")
    a_pat = re.compile(r"^\s*答案[:：]\s*(.*)\s*$")

    def flush():
        nonlocal cur_q, cur_a
        if cur_q is not None and cur_a is not None:
            q = cur_q.strip()
            a = cur_a.strip()
            if q and a:
                qa.append((q, a))
        cur_q, cur_a = None, None

    for t in paras:
        mq = q_pat.match(t)
        ma = a_pat.match(t)

        if mq:
            # New question begins; flush previous pair.
            flush()
            cur_q = mq.group(1).strip()
            continue

        if ma:
            cur_a = ma.group(1).strip()
            continue

        # Continuation lines:
        # Some answers are long and may spill into next paragraph without "答案："
        if cur_a is not None and cur_q is not None:
            cur_a += "\n" + t
        # Some questions might also be split (rare)
        elif cur_q is not None and cur_a is None:
            cur_q += "\n" + t
        else:
            # unrelated heading/code etc
            continue

    flush()
    return qa


def rougeL_f1(pred: str, ref: str, scorer: rouge_scorer.RougeScorer) -> float:
    pred_tokens = " ".join(jieba.lcut(pred))
    ref_tokens = " ".join(jieba.lcut(ref))

    if not pred_tokens.strip() or not ref_tokens.strip():
        return 0.0

    # IMPORTANT: match your provided function: scorer.score(ref_tokens, pred_tokens)
    score = scorer.score(ref_tokens, pred_tokens)
    return float(score["rougeL"].fmeasure)


def strip_question_suffix(answer: str, question: str) -> str:
    """
    Heuristic: if the answer ends with the question (or a long suffix equal to question),
    remove that duplicated suffix to improve Rouge and save tokens.
    """
    a = (answer or "").strip()
    q = (question or "").strip()
    if not a or not q:
        return a

    # direct suffix match
    if a.endswith(q):
        a = a[: -len(q)].rstrip()

    # also handle: "...  矩阵乘法中使用分块技术的优势是什么？"
    # where suffix is q without spaces
    a_nospace = re.sub(r"\s+", "", a)
    q_nospace = re.sub(r"\s+", "", q)
    if a_nospace.endswith(q_nospace):
        # remove last occurrence by finding the last index in original string roughly
        idx = a.rfind(q)
        if idx != -1 and idx > len(a) * 0.5:
            a = a[:idx].rstrip()

    return a


def load_tokenizer(model_dir_or_id: Optional[str]):
    if not model_dir_or_id:
        return None
    if AutoTokenizer is None:
        print("[WARN] transformers not available; token counting will be skipped.")
        return None
    try:
        # fix_mistral_regex only exists for some tokenizers; pass via kwargs safely
        tok = AutoTokenizer.from_pretrained(
            model_dir_or_id,
            trust_remote_code=True,
            use_fast=False,
        )
        return tok
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_dir_or_id, trust_remote_code=True)
        return tok
    except Exception as e:
        print(f"[WARN] failed to load tokenizer from {model_dir_or_id}: {e}")
        return None


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def call_predict(endpoint: str, prompt: str, timeout: int = 300) -> str:
    r = requests.post(endpoint, json={"prompt": prompt}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "response" in data:
        return str(data["response"])
    # fallback
    return json.dumps(data, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--health", default="http://127.0.0.1:8000/")
    ap.add_argument("--basic_docx", default="basic.docx")
    ap.add_argument("--bonus_docx", default="plus.docx")
    ap.add_argument("--which", choices=["basic", "bonus", "all"], default="all")
    ap.add_argument("--max_n", type=int, default=0, help="0 means no limit")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep between requests")
    ap.add_argument("--model_dir_for_tokenizer", default="", help="local path or modelscope/hf id for token counting")
    ap.add_argument("--strip_q_suffix", action="store_true", help="also compute a cleaned score by removing duplicated question suffix")
    ap.add_argument("--save_jsonl", default="eval_details.jsonl")
    args = ap.parse_args()

    # Health check
    try:
        hr = requests.get(args.health, timeout=10)
        print(f"[INFO] health status={hr.status_code}, body={hr.text[:200]}")
    except Exception as e:
        print(f"[WARN] health check failed: {e}")

    tokenizer = load_tokenizer(args.model_dir_for_tokenizer) if args.model_dir_for_tokenizer else None
    if tokenizer is None and args.model_dir_for_tokenizer:
        print("[WARN] tokenizer not loaded; token stats will be 0.")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    datasets: List[Tuple[str, str]] = []
    if args.which in ("basic", "all"):
        datasets.append(("basic", args.basic_docx))
    if args.which in ("bonus", "all"):
        datasets.append(("bonus", args.bonus_docx))

    all_results: List[Dict[str, Any]] = []
    for name, path in datasets:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find {path}. Please pass an absolute path or place it next to this script.")

        qa = parse_docx_qa(path)
        if args.max_n and args.max_n > 0:
            qa = qa[: args.max_n]

        print(f"\n[INFO] Loaded {len(qa)} QA pairs from {name}: {path}")

        total_time = 0.0
        sum_score_raw = 0.0
        sum_score_clean = 0.0
        total_prompt_tokens = 0
        total_output_tokens_raw = 0
        total_output_tokens_clean = 0
        n_ok = 0

        with open(args.save_jsonl, "a", encoding="utf-8") as f:
            for i, (q, ref) in enumerate(tqdm(qa, desc=f"Eval-{name}")):
                t0 = time.perf_counter()
                try:
                    pred = call_predict(args.endpoint, q, timeout=args.timeout)
                    ok = True
                except Exception as e:
                    pred = f"[ERROR] {e}"
                    ok = False
                t1 = time.perf_counter()
                dt = t1 - t0
                total_time += dt

                score_raw = rougeL_f1(pred, ref, scorer) if ok else 0.0
                sum_score_raw += score_raw

                pred_clean = pred
                score_clean = None
                if args.strip_q_suffix and ok:
                    pred_clean = strip_question_suffix(pred, q)
                    score_clean = rougeL_f1(pred_clean, ref, scorer)
                    sum_score_clean += score_clean

                ptok = count_tokens(tokenizer, q)
                otok_raw = count_tokens(tokenizer, pred)
                otok_clean = count_tokens(tokenizer, pred_clean)

                total_prompt_tokens += ptok
                total_output_tokens_raw += otok_raw
                total_output_tokens_clean += otok_clean
                if ok:
                    n_ok += 1

                rec = {
                    "dataset": name,
                    "idx": i,
                    "question": q,
                    "ref": ref,
                    "pred_raw": pred,
                    "pred_clean": pred_clean,
                    "ok": ok,
                    "latency_s": dt,
                    "rougeL_f1_raw": score_raw,
                    "rougeL_f1_clean": score_clean,
                    "prompt_tokens": ptok,
                    "output_tokens_raw": otok_raw,
                    "output_tokens_clean": otok_clean,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                all_results.append(rec)

                if args.sleep > 0:
                    time.sleep(args.sleep)

        denom = len(qa) if len(qa) > 0 else 1
        acc_raw = sum_score_raw / denom
        out_tps_raw = (total_output_tokens_raw / total_time) if total_time > 0 else 0.0
        all_tps_raw = ((total_prompt_tokens + total_output_tokens_raw) / total_time) if total_time > 0 else 0.0

        print(f"\n===== Summary [{name}] =====")
        print(f"Questions: {len(qa)}, OK: {n_ok}, Total time: {total_time:.2f}s")
        print(f"Accuracy (RougeL-F1 mean, RAW): {acc_raw:.4f}")
        print(f"Tokens (prompt/answer RAW): {total_prompt_tokens} / {total_output_tokens_raw}")
        print(f"Throughput RAW: answer_tokens/s={out_tps_raw:.2f}, (prompt+answer)_tokens/s={all_tps_raw:.2f}")

        if args.strip_q_suffix:
            acc_clean = sum_score_clean / denom
            out_tps_clean = (total_output_tokens_clean / total_time) if total_time > 0 else 0.0
            all_tps_clean = ((total_prompt_tokens + total_output_tokens_clean) / total_time) if total_time > 0 else 0.0
            print("\n[With strip_q_suffix]")
            print(f"Accuracy (RougeL-F1 mean, CLEAN): {acc_clean:.4f}")
            print(f"Tokens (prompt/answer CLEAN): {total_prompt_tokens} / {total_output_tokens_clean}")
            print(f"Throughput CLEAN: answer_tokens/s={out_tps_clean:.2f}, (prompt+answer)_tokens/s={all_tps_clean:.2f}")

    print(f"\n[INFO] Detailed records appended to: {args.save_jsonl}")


if __name__ == "__main__":
    main()
