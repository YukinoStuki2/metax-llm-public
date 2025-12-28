import argparse
import json
import os
import re
import time
import random
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
    """解析 docx 中的问答对。

    约定 Q/A 的段落格式为：
      问题：...
      答案：...
    其他标题/段落将被忽略。
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
            # 遇到新问题：先把上一组 Q/A 写入列表。
            flush()
            cur_q = mq.group(1).strip()
            continue

        if ma:
            cur_a = ma.group(1).strip()
            continue

        # 续行处理：
        # 有些答案较长，可能会在没有再次出现“答案：”的情况下延续到下一段。
        if cur_a is not None and cur_q is not None:
            cur_a += "\n" + t
        # 少数情况下问题也可能被拆到多段（较少见）
        elif cur_q is not None and cur_a is None:
            cur_q += "\n" + t
        else:
            # 无关段落（标题/代码块等），忽略
            continue

    flush()
    return qa


def rougeL_f1(pred: str, ref: str, scorer: rouge_scorer.RougeScorer) -> float:
    pred_tokens = " ".join(jieba.lcut(pred))
    ref_tokens = " ".join(jieba.lcut(ref))

    if not pred_tokens.strip() or not ref_tokens.strip():
        return 0.0

    # 重要：保持与评测/提供的实现一致：scorer.score(ref_tokens, pred_tokens)
    score = scorer.score(ref_tokens, pred_tokens)
    return float(score["rougeL"].fmeasure)


def strip_question_suffix(answer: str, question: str) -> str:
    """启发式清理：若答案末尾重复附带了题目文本，则移除该重复后缀。

    目的：提升 Rouge，并减少无意义 token。
    """
    a = (answer or "").strip()
    q = (question or "").strip()
    if not a or not q:
        return a

    # 直接后缀匹配
    if a.endswith(q):
        a = a[: -len(q)].rstrip()

    # 也处理类似：“...  矩阵乘法中使用分块技术的优势是什么？”
    # 即后缀等于去掉空白后的题目
    a_nospace = re.sub(r"\s+", "", a)
    q_nospace = re.sub(r"\s+", "", q)
    if a_nospace.endswith(q_nospace):
        # 通过在原字符串中定位最后一次出现的位置来大致删除
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
        # fix_mistral_regex 仅在少数 tokenizer 存在；这里用 kwargs 方式安全传参
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
    # 兜底：把整个响应序列化为字符串
    return json.dumps(data, ensure_ascii=False)


def call_predict_batch(endpoint: str, prompts: List[str], timeout: int = 300) -> List[str]:
    r = requests.post(endpoint, json={"prompt": prompts}, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and "response" in data:
        resp = data["response"]
        if isinstance(resp, list):
            return ["" if x is None else str(x) for x in resp]
        # 服务端也可能仍返回单个字符串：复制到与 prompts 同长度
        if isinstance(resp, str):
            return [resp] * len(prompts)
        return [str(resp)] * len(prompts)

    # 若服务端直接返回原始 list（非标准协议），也接受。
    if isinstance(data, list):
        return ["" if x is None else str(x) for x in data]

    # 兜底：把整体当作一个大字符串
    return [json.dumps(data, ensure_ascii=False)] * len(prompts)


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
    ap.add_argument("--batch", action="store_true", help="send all questions in one /predict request (prompt=list[str])")
    ap.add_argument("--save_jsonl", default="eval_details.jsonl")
    ap.add_argument(
        "--overwrite_jsonl",
        action="store_true",
        help="overwrite --save_jsonl instead of appending (recommended for repeated experiments)",
    )
    ap.add_argument(
        "--debug_first_n",
        type=int,
        default=0,
        help="print first N (q,pred) pairs and token stats for sanity check",
    )
    ap.add_argument(
        "--debug_random_n",
        type=int,
        default=0,
        help="print N randomly sampled (q,pred) pairs after --debug_first_n (sampled from indices >= debug_first_n)",
    )
    ap.add_argument(
        "--debug_random_seed",
        type=int,
        default=None,
        help="random seed for --debug_random_n. If omitted, a fresh random seed will be used each run (and printed).",
    )
    args = ap.parse_args()

    # 健康检查
    try:
        hr = requests.get(args.health, timeout=10)
        print(f"[INFO] health status={hr.status_code}, body={hr.text[:200]}")
    except Exception as e:
        print(f"[WARN] health check failed: {e}")

    if args.overwrite_jsonl and args.save_jsonl:
        try:
            with open(args.save_jsonl, "w", encoding="utf-8"):
                pass
            print(f"[INFO] overwrite_jsonl=1: truncated {args.save_jsonl}")
        except Exception as e:
            print(f"[WARN] failed to truncate {args.save_jsonl}: {e}")

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
            debug_first_n = max(0, int(args.debug_first_n))
            debug_random_n = max(0, int(args.debug_random_n))
            random_pick: set[int] = set()
            if debug_random_n > 0 and len(qa) > debug_first_n:
                if args.debug_random_seed is None:
                    # 默认：每次运行都随机，同时打印 seed 便于复现。
                    seed = int.from_bytes(os.urandom(8), "little", signed=False)
                    print(f"[INFO] debug_random_seed(auto)={seed}")
                else:
                    seed = int(args.debug_random_seed)
                rng = random.Random(seed)
                # 只从 idx>=debug_first_n 的题中抽，避免与 first_n 重叠
                candidates = list(range(debug_first_n, len(qa)))
                k = min(debug_random_n, len(candidates))
                random_pick = set(rng.sample(candidates, k=k))

            if args.batch:
                qs = [q for (q, _ref) in qa]
                t0 = time.perf_counter()
                try:
                    preds = call_predict_batch(args.endpoint, qs, timeout=args.timeout)
                    ok_all = True
                except Exception as e:
                    preds = [f"[ERROR] {e}"] * len(qs)
                    ok_all = False
                t1 = time.perf_counter()
                dt_total = t1 - t0
                total_time += dt_total

                # Normalize length
                if len(preds) < len(qa):
                    preds = preds + [""] * (len(qa) - len(preds))
                if len(preds) > len(qa):
                    preds = preds[: len(qa)]

                per_item_latency = (dt_total / len(qa)) if len(qa) > 0 else 0.0
                for i, ((q, ref), pred) in enumerate(tqdm(list(zip(qa, preds)), desc=f"Eval-{name}-batch")):
                    ok = ok_all and (not (isinstance(pred, str) and pred.startswith("[ERROR]")))

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

                    if (debug_first_n > 0 and i < debug_first_n) or (i in random_pick):
                        p_preview = (pred or "").replace("\n", "\\n")
                        if len(p_preview) > 240:
                            p_preview = p_preview[:240] + "..."
                        print(f"\n[DEBUG] {name}[{i}] q={q}")
                        print(f"[DEBUG] {name}[{i}] pred_preview={p_preview}")
                        if ok:
                            if args.strip_q_suffix:
                                print(
                                    f"[DEBUG] {name}[{i}] rougeL_f1_raw={score_raw:.4f}, rougeL_f1_clean={float(score_clean or 0.0):.4f}"
                                )
                            else:
                                print(f"[DEBUG] {name}[{i}] rougeL_f1_raw={score_raw:.4f}")
                        else:
                            print(f"[DEBUG] {name}[{i}] rougeL_f1_raw=0.0000 (request failed)")
                        if args.model_dir_for_tokenizer:
                            print(f"[DEBUG] {name}[{i}] tokens(prompt/question)={ptok}, tokens(output)={otok_raw}")

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
                        "latency_s": per_item_latency,
                        "rougeL_f1_raw": score_raw,
                        "rougeL_f1_clean": score_clean,
                        "prompt_tokens": ptok,
                        "output_tokens_raw": otok_raw,
                        "output_tokens_clean": otok_clean,
                        "batch_total_latency_s": dt_total,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    all_results.append(rec)
            else:
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

                    if (debug_first_n > 0 and i < debug_first_n) or (i in random_pick):
                        p_preview = (pred or "").replace("\n", "\\n")
                        if len(p_preview) > 240:
                            p_preview = p_preview[:240] + "..."
                        print(f"\n[DEBUG] {name}[{i}] q={q}")
                        print(f"[DEBUG] {name}[{i}] pred_preview={p_preview}")
                        if ok:
                            if args.strip_q_suffix:
                                print(
                                    f"[DEBUG] {name}[{i}] rougeL_f1_raw={score_raw:.4f}, rougeL_f1_clean={float(score_clean or 0.0):.4f}"
                                )
                            else:
                                print(f"[DEBUG] {name}[{i}] rougeL_f1_raw={score_raw:.4f}")
                        else:
                            print(f"[DEBUG] {name}[{i}] rougeL_f1_raw=0.0000 (request failed)")
                        if args.model_dir_for_tokenizer:
                            print(f"[DEBUG] {name}[{i}] tokens(prompt/question)={ptok}, tokens(output)={otok_raw}")

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

        if args.model_dir_for_tokenizer and total_output_tokens_raw == 0:
            print("[WARN] output token count is 0. Throughput numbers may be meaningless; tokenizer may be incompatible.")

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
