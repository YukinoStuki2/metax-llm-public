#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""自动化调参（守护式）脚本。

目标：
- 每个参数组合：启动服务 -> 健康检查 -> 跑 N 次 eval_local.py -> 取平均 -> 记录 -> 关闭服务
- 失败/崩溃/超时：记录失败并自动进入下一轮（主脚本不退出）
- 断点续跑：结果写入 JSONL，重启脚本会跳过已完成组合

不依赖额外第三方库（仅用标准库）。
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import ssl
import smtplib
from email.message import EmailMessage
import urllib.request
import base64
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Tuple, Set


ACCURACY_RE = re.compile(r"Accuracy \(RougeL-F1 mean, RAW\):\s*([0-9.]+)")
ANSWER_TPS_RE = re.compile(r"Throughput RAW:.*answer_tokens/s=([0-9.]+)")
TOTAL_TPS_RE = re.compile(r"\(prompt\+answer\)_tokens/s=([0-9.]+)")
TOTAL_TIME_RE = re.compile(r"Total time:\s*([0-9.]+)s")


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def repo_root_from_script() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_json_file(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def merge_search_space(base: Dict[str, List[str]], extra: Dict[str, Any]) -> Dict[str, List[str]]:
    """合并外部搜索空间。

    extra 格式示例：
      {"GPU_MEMORY_UTILIZATION": ["0.965", "0.975"], "NEW_PARAM": ["a", "b"]}
    """
    out: Dict[str, List[str]] = {k: list(v) for k, v in (base or {}).items()}
    for k, vv in (extra or {}).items():
        if vv is None:
            continue
        if isinstance(vv, (list, tuple)):
            items = [str(x) for x in vv]
        else:
            items = [str(vv)]
        cur = out.get(str(k), [])
        seen = set(cur)
        for x in items:
            if x not in seen:
                cur.append(x)
                seen.add(x)
        out[str(k)] = cur
    return out


def _list_listening_pids(port: int) -> Set[int]:
    """尽量从 ss 输出中解析监听指定端口的 pid 集合。

    说明：容器里通常有 ss；没有的话返回空集合。
    """
    try:
        cp = subprocess.run(
            ["ss", "-ltnp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
        )
        out = cp.stdout or ""
    except Exception:
        return set()

    pids: Set[int] = set()
    for line in out.splitlines():
        if f":{port}" not in line:
            continue
        for m in re.finditer(r"pid=(\d+)", line):
            try:
                pids.add(int(m.group(1)))
            except Exception:
                pass
    return pids


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
    except Exception:
        return ""


def try_free_port(port: int, wait_s: float, kill: bool) -> bool:
    """尝试释放端口：先等一等；必要时（可选）安全地杀掉明显的残留 uvicorn serve:app 进程。"""
    if ensure_port_free(port, wait_s=0.5):
        return True

    if kill:
        pids = _list_listening_pids(port)
        # 只杀“很像我们服务”的进程，避免误伤。
        candidates: List[int] = []
        for pid in sorted(pids):
            cmd = _read_cmdline(pid)
            if not cmd:
                continue
            if "uvicorn" in cmd and "serve:app" in cmd:
                candidates.append(pid)

        for pid in candidates:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        # 给一点时间优雅退出
        deadline = time.time() + min(5.0, max(0.5, float(wait_s)))
        while time.time() < deadline:
            if ensure_port_free(port, wait_s=0.5):
                return True
            time.sleep(0.5)

        # 还没释放，再强杀
        for pid in candidates:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    # 最后再等 wait_s
    return ensure_port_free(port, wait_s=float(wait_s))


def try_send_webhook(url: str, payload: Dict[str, Any], timeout_s: float = 8.0) -> Optional[str]:
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return f"{resp.status} {resp.reason}"
    except Exception as e:
        return f"webhook error: {type(e).__name__}: {e}"


def _feishu_sign(timestamp: str, secret: str) -> str:
    # 飞书自定义机器人签名：base64(hmac_sha256(f"{timestamp}\n{secret}", key=secret))
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    h = hmac.new(secret.encode("utf-8"), string_to_sign, hashlib.sha256).digest()
    return base64.b64encode(h).decode("utf-8")


def try_send_feishu_webhook(
    url: str,
    title: str,
    text: str,
    secret: str = "",
    timeout_s: float = 8.0,
) -> Optional[str]:
    """飞书群自定义机器人 webhook。

    - 默认发送交互卡片（信息密度更高）
    - 若启用签名安全（secret），会自动附带 timestamp/sign
    """
    try:
        ts = str(int(time.time()))
        body: Dict[str, Any] = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": title}},
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": text}},
                ],
            },
        }
        if secret:
            body["timestamp"] = ts
            body["sign"] = _feishu_sign(ts, secret)

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            # 飞书 webhook：HTTP 200 不代表成功，需看 JSON 里的 code
            try:
                obj = json.loads(raw) if raw else {}
            except Exception:
                obj = {}
            if isinstance(obj, dict) and ("code" in obj):
                code = obj.get("code")
                if code == 0:
                    return None
                msg = obj.get("msg") or obj.get("message") or "unknown"
                return f"feishu code={code} msg={msg}"
            # 兜底：若无标准 JSON，则仅以 HTTP 状态判断
            if 200 <= int(getattr(resp, "status", 0) or 0) < 300:
                return None
            return f"http {resp.status} {resp.reason} body={raw[:200]}"
    except Exception as e:
        return f"feishu error: {type(e).__name__}: {e}"


def try_send_email_smtp(
    host: str,
    port: int,
    user: str,
    password: str,
    mail_from: str,
    mail_to: str,
    subject: str,
    body: str,
    timeout_s: float = 12.0,
    use_starttls: bool = True,
    use_ssl: bool = False,
) -> Optional[str]:
    try:
        msg = EmailMessage()
        msg["From"] = mail_from
        msg["To"] = mail_to
        msg["Subject"] = subject
        msg.set_content(body)

        if use_ssl:
            with smtplib.SMTP_SSL(host=host, port=port, timeout=timeout_s) as s:
                s.ehlo()
                if user:
                    s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host=host, port=port, timeout=timeout_s) as s:
                s.ehlo()
                if use_starttls:
                    ctx = ssl.create_default_context()
                    s.starttls(context=ctx)
                    s.ehlo()
                if user:
                    s.login(user, password)
                s.send_message(msg)
        return None
    except Exception as e:
        return f"smtp error: {type(e).__name__}: {e}"


def compact_params(params: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in params:
            out[k] = params[k]
    return out


def is_port_open(host: str, port: int, timeout_s: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def http_get_localhost(path: str, port: int = 8000, timeout_s: float = 1.0) -> Tuple[int, str]:
    """最小 HTTP GET（避免引入 requests）。"""
    host = "127.0.0.1"
    with socket.create_connection((host, port), timeout=timeout_s) as s:
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        s.sendall(req.encode("utf-8"))
        s.settimeout(timeout_s)
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    # 简单解析 status code
    head = data.split(b"\r\n\r\n", 1)[0].decode("utf-8", errors="ignore")
    m = re.search(r"HTTP/\d\.\d\s+(\d+)", head)
    code = int(m.group(1)) if m else 0
    body = ""
    if b"\r\n\r\n" in data:
        body = data.split(b"\r\n\r\n", 1)[1].decode("utf-8", errors="ignore")
    return code, body[:500]


def wait_server_ready(
    port: int,
    timeout_s: int,
    poll_interval_s: float = 0.5,
) -> Tuple[bool, str]:
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        if not is_port_open("127.0.0.1", port):
            time.sleep(poll_interval_s)
            continue
        try:
            code, body = http_get_localhost("/", port=port, timeout_s=1.0)
            if 200 <= code < 500:
                return True, f"health={code} body={body!r}"
            last_err = f"bad status={code} body={body!r}"
        except Exception as e:
            last_err = f"health exception: {type(e).__name__}: {e}"
        time.sleep(poll_interval_s)
    return False, last_err or "timeout"


def kill_process_group(p: subprocess.Popen, grace_s: float = 10.0) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        try:
            p.terminate()
        except Exception:
            return

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if p.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def parse_eval_output(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    acc = None
    answer_tps = None
    total_tps = None
    total_time_s = None
    m1 = ACCURACY_RE.search(text)
    if m1:
        try:
            acc = float(m1.group(1))
        except Exception:
            acc = None
    m2 = ANSWER_TPS_RE.search(text)
    if m2:
        try:
            answer_tps = float(m2.group(1))
        except Exception:
            answer_tps = None
    m3 = TOTAL_TPS_RE.search(text)
    if m3:
        try:
            total_tps = float(m3.group(1))
        except Exception:
            total_tps = None
    m4 = TOTAL_TIME_RE.search(text)
    if m4:
        try:
            total_time_s = float(m4.group(1))
        except Exception:
            total_time_s = None
    return acc, answer_tps, total_tps, total_time_s


@dataclasses.dataclass(frozen=True)
class Trial:
    params: Dict[str, str]

    def key(self) -> str:
        items = sorted(self.params.items(), key=lambda x: x[0])
        return "|".join([f"{k}={v}" for k, v in items])


def load_completed_keys(jsonl_path: str) -> set[str]:
    if not os.path.exists(jsonl_path):
        return set()
    done: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get("trial_key")
            status = obj.get("status")
            if isinstance(k, str) and status in ("ok", "failed", "crashed", "timeout"):
                done.add(k)
    return done


def ensure_port_free(port: int, wait_s: float = 8.0) -> bool:
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if not is_port_open("127.0.0.1", port):
            return True
        time.sleep(0.2)
    return not is_port_open("127.0.0.1", port)


def build_candidate_values(current_value: Optional[str], candidates: List[str]) -> List[str]:
    uniq: List[str] = []
    if current_value is not None:
        uniq.append(str(current_value))
    for v in candidates:
        v = str(v)
        if v not in uniq:
            uniq.append(v)
    return uniq


def choose_best(
    results: List[Dict[str, Any]],
    acc_threshold: float,
) -> Optional[Dict[str, Any]]:
    """在一组（同一参数名的）试验结果中选最优：
    - 先过滤 acc >= threshold
    - 再按 (prompt+answer)_tokens/s 降序（你更关心的指标）
    - 缺失视为 -inf
    """
    ok = [r for r in results if r.get("status") == "ok" and (r.get("avg_accuracy") is not None)]
    ok = [r for r in ok if float(r["avg_accuracy"]) >= acc_threshold]
    if not ok:
        return None
    def score(r: Dict[str, Any]) -> float:
        thr = r.get("avg_total_tps")
        try:
            return float(thr)
        except Exception:
            return float("-inf")
    ok.sort(key=score, reverse=True)
    return ok[0]


def _fmt_duration(seconds: float) -> str:
    try:
        s = int(seconds)
    except Exception:
        s = 0
    if s < 0:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=repo_root_from_script(), help="仓库路径（默认脚本所在目录）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--eval_runs", type=int, default=5)
    ap.add_argument("--startup_timeout", type=int, default=240)
    ap.add_argument("--eval_timeout", type=int, default=420)
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--accuracy_threshold", type=float, default=0.8800)
    ap.add_argument("--results", default="tune_results.jsonl")
    ap.add_argument("--best", default="best_params.json")
    ap.add_argument("--server_log_dir", default="tune_server_logs")
    ap.add_argument("--status_file", default="tune_status.json", help="实时状态输出（json），便于外部查看")
    ap.add_argument("--heartbeat_trials", type=int, default=10, help="每 N 个 trial 发一次心跳通知（0=关闭）")
    ap.add_argument(
        "--heartbeat_interval_s",
        type=int,
        default=int(os.environ.get("TUNE_HEARTBEAT_INTERVAL_S", "0") or "0"),
        help="每隔 N 秒发一次心跳通知（0=关闭）。例如 600=10min",
    )

    ap.add_argument(
        "--search_space_file",
        default=os.environ.get("TUNE_SEARCH_SPACE_FILE", ""),
        help="外部搜索空间 JSON 文件路径（可选）。格式：{\"PARAM\":[\"v1\",\"v2\"], ... }",
    )

    ap.add_argument(
        "--port_busy_retries",
        type=int,
        default=int(os.environ.get("TUNE_PORT_BUSY_RETRIES", "3") or "3"),
        help="端口占用时重试次数（每次会等待/尝试释放）。",
    )
    ap.add_argument(
        "--port_busy_wait_s",
        type=float,
        default=float(os.environ.get("TUNE_PORT_BUSY_WAIT_S", "10") or "10"),
        help="端口占用时每次重试的等待秒数。",
    )
    ap.add_argument(
        "--port_busy_kill",
        action="store_true",
        default=(os.environ.get("TUNE_PORT_BUSY_KILL", "0") == "1"),
        help="端口占用时尝试清理残留 uvicorn serve:app（谨慎开关）。",
    )

    ap.add_argument(
        "--notify_trial_done",
        action="store_true",
        default=(os.environ.get("TUNE_NOTIFY_TRIAL_DONE", "0") == "1"),
        help="每个 trial 完成后都发一次通知（含准确率/速度/耗时）；默认关闭，避免刷屏",
    )
    ap.add_argument(
        "--notify_trial_done_every",
        type=int,
        default=int(os.environ.get("TUNE_NOTIFY_TRIAL_DONE_EVERY", "1") or "1"),
        help="每 N 个 trial 发一次 trial_done 通知（配合 --notify_trial_done 使用）",
    )

    # 通知：SMTP 邮件（可选）
    ap.add_argument("--smtp_host", default=os.environ.get("TUNE_SMTP_HOST", ""))
    ap.add_argument("--smtp_port", type=int, default=int(os.environ.get("TUNE_SMTP_PORT", "587")))
    ap.add_argument("--smtp_user", default=os.environ.get("TUNE_SMTP_USER", ""))
    ap.add_argument("--smtp_pass", default=os.environ.get("TUNE_SMTP_PASS", ""))
    ap.add_argument("--smtp_from", default=os.environ.get("TUNE_SMTP_FROM", ""))
    ap.add_argument("--smtp_to", default=os.environ.get("TUNE_SMTP_TO", ""))
    ap.add_argument("--smtp_no_starttls", action="store_true")
    ap.add_argument("--smtp_ssl", action="store_true", help="使用 SMTPS（如 465 端口常用）")
    ap.add_argument(
        "--email_kinds",
        default=os.environ.get("TUNE_EMAIL_KINDS", "best,crashed,done"),
        help="哪些事件发邮件（逗号分隔）；默认仅兜底关键事件",
    )

    # 通知：Webhook（可选）
    ap.add_argument("--webhook_url", default=os.environ.get("TUNE_WEBHOOK_URL", ""))
    ap.add_argument("--feishu_secret", default=os.environ.get("TUNE_FEISHU_SECRET", ""), help="飞书机器人安全设置的签名 secret（可选）")
    ap.add_argument(
        "--webhook_kind",
        choices=["generic", "feishu"],
        default=os.environ.get("TUNE_WEBHOOK_KIND", "feishu"),
        help="webhook 类型：generic=普通 JSON POST；feishu=飞书自定义机器人",
    )
    ap.add_argument("--skip_existing", action="store_true", help="跳过 results 里已完成的 trial")
    ap.add_argument("--dry_run", action="store_true", help="只生成 trial 列表并退出（不启动模型）")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="不启动模型的自检：验证解析/落盘/断点/（可选）通知是否可用",
    )
    ap.add_argument(
        "--selftest_notify",
        action="store_true",
        help="在 selftest 时也发送通知（飞书/邮件，若已配置）",
    )

    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    results_path = os.path.join(repo, args.results)
    best_path = os.path.join(repo, args.best)
    server_log_dir = os.path.join(repo, args.server_log_dir)
    status_path = os.path.join(repo, args.status_file)

    run_started_ts = time.time()
    last_heartbeat_ts = 0.0

    def uptime_str() -> str:
        return _fmt_duration(time.time() - run_started_ts)
    os.makedirs(server_log_dir, exist_ok=True)

    current_server: Dict[str, Optional[subprocess.Popen]] = {"p": None}

    def _handle_signal(signum: int, _frame: Any) -> None:
        p = current_server.get("p")
        if isinstance(p, subprocess.Popen):
            try:
                kill_process_group(p)
            except Exception:
                pass
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def write_status(obj: Dict[str, Any]) -> None:
        try:
            obj = dict(obj)
            obj.setdefault("ts", now_iso())
            atomic_write_json(status_path, obj)
        except Exception:
            pass

    def notify(kind: str, title: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "kind": kind,
            "title": title,
            "message": message,
            "ts": now_iso(),
            "repo": repo,
        }
        if extra:
            payload["extra"] = extra

        # webhook（飞书优先）
        if args.webhook_url:
            if args.webhook_kind == "feishu":
                md = (
                    f"**事件**：{kind}\n"
                    f"**时间**：{payload['ts']}\n"
                    f"**已运行**：{uptime_str()}\n"
                    f"**信息**：{message}\n"
                    f"**结果文件**：`{os.path.relpath(results_path, repo)}`\n"
                    f"**最优参数**：`{os.path.relpath(best_path, repo)}`\n"
                    f"**状态文件**：`{os.path.relpath(status_path, repo)}`\n"
                )
                if extra:
                    try:
                        md += "\n**附加信息**：\n" + "```\n" + json.dumps(extra, ensure_ascii=False, indent=2) + "\n```\n"
                    except Exception:
                        pass
                try_send_feishu_webhook(args.webhook_url, title=title, text=md, secret=args.feishu_secret)
            else:
                try_send_webhook(args.webhook_url, payload)

        # email（兜底：默认仅 best/crashed/done）
        email_kinds = {x.strip() for x in str(args.email_kinds).split(",") if x.strip()}
        if args.smtp_host and args.smtp_to and (kind in email_kinds):
            mail_from = args.smtp_from or args.smtp_user or args.smtp_to
            err = try_send_email_smtp(
                host=args.smtp_host,
                port=int(args.smtp_port),
                user=args.smtp_user,
                password=args.smtp_pass,
                mail_from=mail_from,
                mail_to=args.smtp_to,
                subject=f"[auto_tune] {title}",
                body=json.dumps(payload, ensure_ascii=False, indent=2),
                use_starttls=(not args.smtp_no_starttls),
                use_ssl=bool(args.smtp_ssl),
            )
            if err:
                pass

    def run_selftest() -> int:
        # 1) 解析样例输出
        sample = (
            "===== Summary [bonus] =====\n"
            "Questions: 92, OK: 92, Total time: 1.76s\n"
            "Accuracy (RougeL-F1 mean, RAW): 0.8815\n"
            "Tokens (prompt/answer RAW): 1907 / 8554\n"
            "Throughput RAW: answer_tokens/s=4857.52, (prompt+answer)_tokens/s=5940.43\n"
        )
        acc, ans_tps, total_tps, total_time_s = parse_eval_output(sample)
        if not (acc and abs(acc - 0.8815) < 1e-6):
            print("[selftest] parse accuracy FAILED", acc)
            return 2
        if not (total_tps and abs(total_tps - 5940.43) < 1e-2):
            print("[selftest] parse total_tps FAILED", total_tps)
            return 2
        if not (ans_tps and abs(ans_tps - 4857.52) < 1e-2):
            print("[selftest] parse answer_tps FAILED", ans_tps)
            return 2
        if not (total_time_s and abs(total_time_s - 1.76) < 1e-3):
            print("[selftest] parse total_time FAILED", total_time_s)
            return 2

        # 2) 写状态文件/结果文件（不覆盖真实默认文件：建议用户用 --results/--status_file 指定）
        write_status({"phase": "selftest", "note": "status write ok"})
        rec = {
            "ts": now_iso(),
            "trial_key": "selftest",
            "status": "ok",
            "avg_accuracy": acc,
            "avg_total_tps": total_tps,
            "avg_answer_tps": ans_tps,
            "avg_total_time_s": total_time_s,
            "params": {"SELFTEST": "1"},
        }
        append_jsonl(results_path, rec)
        done_keys = load_completed_keys(results_path)
        if "selftest" not in done_keys:
            print("[selftest] load_completed_keys FAILED")
            return 2

        # 3) best 选择逻辑
        r1 = dict(rec)
        r1["trial_key"] = "r1"
        r2 = dict(rec)
        r2["trial_key"] = "r2"
        r2["avg_total_tps"] = float(rec["avg_total_tps"]) + 10.0
        best = choose_best([r1, r2], acc_threshold=float(args.accuracy_threshold))
        if not best or best.get("trial_key") != "r2":
            print("[selftest] choose_best FAILED", best)
            return 2

        if args.selftest_notify:
            notify("heartbeat", "selftest", "selftest notification", {"ok": True})

        print("[selftest] OK")
        return 0

    # 固定 eval 指令（按你的要求）
    # 注意：这里用 sys.executable，避免不同 python 环境导致行为不一致。
    eval_cmd = [
        sys.executable,
        "eval_local.py",
        "--which",
        "bonus",
        "--model_dir_for_tokenizer",
        "./model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM",
        "--batch",
        "--overwrite_jsonl",
    ]

    # 读入当前 best（若不存在则用 env_force 默认值）
    base_params: Dict[str, str] = {}
    if os.path.exists(best_path):
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                base_params = json.load(f) or {}
        except Exception:
            base_params = {}

    if not base_params:
        # 兜底：与 env_force.sh 对齐的一些默认值
        base_params = {
            "MAX_NEW_TOKENS": "64",
            "MAX_NEW_TOKENS_CODE": "192",
            "MAX_NEW_TOKENS_CODE_HARD": "192",
            "MAX_NEW_TOKENS_CODE_SOFT": "96",
            "HARD_CODE_MIN_HITS": "1",
            "LONG_ANSWER_ENABLE_DEFAULT": "1",
            "LONG_ANSWER_MIN_HITS": "1",
            "REPETITION_PENALTY": "1.05",
            "FREQUENCY_PENALTY": "0.1",
            "STOP_STRINGS": "<|im_end|>,<|endoftext|>",
            "STOP_ON_DOUBLE_NEWLINE": "0",
            "GPU_MEMORY_UTILIZATION": "0.97",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "131072",
            "VLLM_MAX_NUM_SEQS": "1024",
            "MAX_MODEL_LEN": "1024",
            "OUTPUT_TRIM_EXAMPLES": "1",
            "OUTPUT_MAX_SENTENCES": "6",
        }

    # 搜索空间：尽量覆盖你列出的参数，但保持“先单参坐标上升”的高效率
    search_space: Dict[str, List[str]] = {
        # 显存不需要逼近极限：优先测 0.97/0.98/0.99（更稳定）
        "GPU_MEMORY_UTILIZATION": ["0.97", "0.98", "0.99"],
        "VLLM_MAX_NUM_BATCHED_TOKENS": ["65536", "98304", "131072", "196608", "262144"],
        "VLLM_MAX_NUM_SEQS": ["512", "768", "1024", "1536"],
        "MAX_MODEL_LEN": ["768", "1024", "1536", "2048"],
        "STOP_ON_DOUBLE_NEWLINE": ["0", "1"],
        "STOP_STRINGS": [
            "<|im_end|>,<|endoftext|>",
            "<|im_end|>",
            "<|endoftext|>",
        ],
        "MAX_NEW_TOKENS": ["32", "48", "64", "80"],
        "MAX_NEW_TOKENS_CODE": ["128", "160", "192", "256"],
        "MAX_NEW_TOKENS_CODE_SOFT": ["64", "96", "128"],
        "MAX_NEW_TOKENS_CODE_HARD": ["192", "256", "320"],
        "HARD_CODE_MIN_HITS": ["1", "2"],
        "LONG_ANSWER_ENABLE_DEFAULT": ["0", "1"],
        "LONG_ANSWER_MIN_HITS": ["1", "2", "3"],
        "REPETITION_PENALTY": ["1.00", "1.02", "1.05"],
        "FREQUENCY_PENALTY": ["0.00", "0.05", "0.10"],
        "OUTPUT_TRIM_EXAMPLES": ["0", "1"],
        "OUTPUT_MAX_SENTENCES": ["3", "4", "6", "8"],
    }

    # 可选：从外部 JSON 合并搜索空间，便于在“合理范围内”扩展组合数量。
    # 格式：{"PARAM":["v1","v2"], "NEW_PARAM":["a","b"]}
    if args.search_space_file:
        extra = _load_json_file(args.search_space_file)
        if isinstance(extra, dict):
            search_space = merge_search_space(search_space, extra)

    order = [
        # 先找一个不 OOM 的高利用率
        "GPU_MEMORY_UTILIZATION",
        # 再压榨吞吐
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_MAX_NUM_SEQS",
        "MAX_MODEL_LEN",
        # 再看停止条件与输出裁剪
        "STOP_ON_DOUBLE_NEWLINE",
        "STOP_STRINGS",
        "OUTPUT_TRIM_EXAMPLES",
        "OUTPUT_MAX_SENTENCES",
        # 再调分流/长度相关
        "MAX_NEW_TOKENS",
        "MAX_NEW_TOKENS_CODE",
        "MAX_NEW_TOKENS_CODE_SOFT",
        "MAX_NEW_TOKENS_CODE_HARD",
        "HARD_CODE_MIN_HITS",
        "LONG_ANSWER_ENABLE_DEFAULT",
        "LONG_ANSWER_MIN_HITS",
        # 最后试试惩罚项（可能影响准确率）
        "REPETITION_PENALTY",
        "FREQUENCY_PENALTY",
    ]

    # 外部 search_space 里如果带了新参数名，默认追加到最后探索。
    for k in search_space.keys():
        if k not in order:
            order.append(k)

    done = load_completed_keys(results_path) if args.skip_existing else set()

    if args.selftest:
        return run_selftest()

    # dry-run：估算总试验数并打印前若干条
    if args.dry_run:
        total = 0
        preview: List[str] = []
        cur = dict(base_params)
        for name in order:
            cand = search_space.get(name, [])
            if not cand:
                continue
            vals = build_candidate_values(cur.get(name), cand)
            total += len(vals)
            for v in vals:
                p = dict(cur)
                p[name] = v
                k = Trial(params=p).key()
                if len(preview) < 50:
                    preview.append(k)
            # dry-run 不更新 cur（因为不跑结果），仅用于估算。
        print(f"[dry_run] trials~={total}")
        for k in preview:
            print(k)
        if total > len(preview):
            print("...")
        return 0

    trial_index = 0

    # 全局最优追踪：用你关心的 total_tps
    best_seen_total_tps: float = float("-inf")
    if os.path.exists(best_path):
        # best_path 只存 params，不存分数；所以 best_seen_total_tps 仍从 -inf 开始
        pass

    write_status({"phase": "start", "best_params": base_params, "results": os.path.relpath(results_path, repo)})
    notify("start", "tuning started", "auto_tune started", {"accuracy_threshold": args.accuracy_threshold})
    last_heartbeat_ts = time.time()

    # 核心：按参数逐组测试；每组结束就更新 base_params（实时保持最优参数）。
    for param_name in order:
        candidates = search_space.get(param_name, [])
        if not candidates:
            continue

        group_results: List[Dict[str, Any]] = []
        values = build_candidate_values(base_params.get(param_name), candidates)
        write_status({"phase": "param_group", "param": param_name, "base_params": base_params})

        for v in values:
            trial_index += 1
            trial_params = dict(base_params)
            trial_params[param_name] = str(v)
            trial = Trial(params=trial_params)
            trial_key = trial.key()

            write_status(
                {
                    "phase": "trial_start",
                    "trial_index": trial_index,
                    "trial_key": trial_key,
                    "param": param_name,
                    "params": trial.params,
                    "best_params": base_params,
                }
            )

            # 时间心跳：每隔固定秒数发一次（更适合长时间运行/容器保活观察）
            if args.heartbeat_interval_s and args.heartbeat_interval_s > 0:
                now_ts = time.time()
                if (now_ts - last_heartbeat_ts) >= float(args.heartbeat_interval_s):
                    last_heartbeat_ts = now_ts
                    notify(
                        "heartbeat",
                        "heartbeat",
                        f"alive; uptime={uptime_str()}; trial={trial_index}; param={param_name} value={v}",
                        {"trial_key": trial_key, "param": param_name, "trial_index": trial_index},
                    )

            if args.heartbeat_trials and args.heartbeat_trials > 0 and (trial_index % int(args.heartbeat_trials) == 0):
                notify(
                    "heartbeat",
                    "heartbeat",
                    f"trial {trial_index} running; param={param_name} value={v}",
                    {"trial_key": trial_key, "param": param_name},
                )

            if args.skip_existing and trial_key in done:
                continue

            # 端口占用：不要直接连续失败（会导致“探索数量看起来很多但全是失败”）。
            # 这里做带重试的等待/（可选）清理残留进程。
            port_ok = False
            for attempt in range(max(0, int(args.port_busy_retries)) + 1):
                if try_free_port(args.port, wait_s=float(args.port_busy_wait_s), kill=bool(args.port_busy_kill)):
                    port_ok = True
                    break
                write_status(
                    {
                        "phase": "port_busy",
                        "trial_index": trial_index,
                        "trial_key": trial_key,
                        "param": param_name,
                        "attempt": attempt + 1,
                        "message": f"port {args.port} still in use",
                    }
                )
            if not port_ok:
                append_jsonl(
                    results_path,
                    {
                        "ts": now_iso(),
                        "trial_key": trial_key,
                        "status": "failed",
                        "reason": f"port {args.port} still in use before start",
                        "params": trial.params,
                    },
                )
                notify("failed", "port busy", f"port {args.port} still in use", {"trial_key": trial_key})
                continue

            # 每轮单独日志
            log_path = os.path.join(server_log_dir, f"trial_{trial_index:05d}.log")

            env = os.environ.copy()
            env.update({k: str(vv) for k, vv in trial.params.items()})
            # 为调参加速：默认跳过每轮 pip / download（你也可以手动 export 覆盖为 0）
            env.setdefault("SKIP_PIP_INSTALL", "1")
            env.setdefault("SKIP_MODEL_DOWNLOAD", "1")

            # 启动 run_model.sh（独立会话/进程组，避免 shell 崩溃影响主脚本）
            start_ts = time.time()
            with open(log_path, "wb") as logf:
                p = subprocess.Popen(
                    ["bash", "run_model.sh"],
                    cwd=repo,
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                current_server["p"] = p

                ok, health_msg = wait_server_ready(args.port, timeout_s=int(args.startup_timeout))
                if not ok:
                    status = "timeout" if p.poll() is None else "crashed"
                    kill_process_group(p)
                    rec = {
                        "ts": now_iso(),
                        "trial_key": trial_key,
                        "status": status,
                        "phase": "startup",
                        "param_under_test": param_name,
                        "health": health_msg,
                        "exit_code": p.poll(),
                        "server_log": os.path.relpath(log_path, repo),
                        "params": trial.params,
                        "startup_s": round(time.time() - start_ts, 3),
                    }
                    append_jsonl(results_path, rec)
                    group_results.append(rec)
                    done.add(trial_key)
                    notify(status, f"startup {status}", f"param={param_name} value={v} health={health_msg}", {
                        "trial_key": trial_key,
                        "server_log": rec.get("server_log"),
                    })
                    time.sleep(args.cooldown)
                    continue

                # 评测 N 次
                eval_runs: List[Dict[str, Any]] = []
                sum_acc = 0.0
                sum_thr = 0.0
                ok_n = 0
                for r_i in range(1, int(args.eval_runs) + 1):
                    t0 = time.time()
                    try:
                        cp = subprocess.run(
                            eval_cmd,
                            cwd=repo,
                            env=os.environ.copy(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            timeout=int(args.eval_timeout),
                            text=True,
                        )
                        out = cp.stdout or ""
                        acc, answer_tps, total_tps, total_time_s = parse_eval_output(out)
                        eval_runs.append(
                            {
                                "i": r_i,
                                "ok": cp.returncode == 0 and acc is not None and total_tps is not None,
                                "returncode": cp.returncode,
                                "accuracy": acc,
                                "answer_tps": answer_tps,
                                "total_tps": total_tps,
                                "total_time_s": total_time_s,
                                "elapsed_s": round(time.time() - t0, 3),
                            }
                        )
                        if cp.returncode == 0 and acc is not None and total_tps is not None:
                            sum_acc += float(acc)
                            sum_thr += float(total_tps)
                            ok_n += 1
                    except subprocess.TimeoutExpired:
                        eval_runs.append({"i": r_i, "ok": False, "timeout": True})
                    except Exception as e:
                        eval_runs.append({"i": r_i, "ok": False, "error": f"{type(e).__name__}: {e}"})

                # 关服（无论评测成功与否）
                kill_process_group(p)
                ensure_port_free(args.port, wait_s=8.0)
                current_server["p"] = None

            avg_acc = (sum_acc / ok_n) if ok_n > 0 else None
            avg_total_tps = (sum_thr / ok_n) if ok_n > 0 else None

            # 同时也算一个平均 answer_tokens/s（仅用于参考显示）
            sum_answer = 0.0
            ans_n = 0
            sum_total_time = 0.0
            time_n = 0
            for rr in eval_runs:
                if rr.get("ok") and rr.get("answer_tps") is not None:
                    try:
                        sum_answer += float(rr["answer_tps"])
                        ans_n += 1
                    except Exception:
                        pass
                if rr.get("ok") and rr.get("total_time_s") is not None:
                    try:
                        sum_total_time += float(rr["total_time_s"])
                        time_n += 1
                    except Exception:
                        pass
            avg_answer_tps = (sum_answer / ans_n) if ans_n > 0 else None
            avg_total_time_s = (sum_total_time / time_n) if time_n > 0 else None

            rec = {
                "ts": now_iso(),
                "trial_key": trial_key,
                "status": "ok" if ok_n == int(args.eval_runs) else ("failed" if ok_n == 0 else "partial"),
                "phase": "eval",
                "param_under_test": param_name,
                "params": trial.params,
                "health": health_msg,
                "eval_ok_n": ok_n,
                "eval_runs": eval_runs,
                "avg_accuracy": avg_acc,
                "avg_total_tps": avg_total_tps,
                "avg_answer_tps": avg_answer_tps,
                "avg_total_time_s": avg_total_time_s,
                "server_log": os.path.relpath(log_path, repo),
                "total_s": round(time.time() - start_ts, 3),
            }
            append_jsonl(results_path, rec)
            group_results.append(rec)
            done.add(trial_key)

            # 你提到需要“确切证据（本轮准确率/速度/失败）再开始下一轮”。
            # 脚本本身就是串行跑：一轮 eval 结束并写入 rec 后才会进入下一轮。
            # 这里提供可选的 per-trial 通知，把本轮结果直接推到飞书。
            if args.notify_trial_done:
                every = max(1, int(args.notify_trial_done_every))
                if (trial_index % every) == 0:
                    a = rec.get("avg_accuracy")
                    tps = rec.get("avg_total_tps")
                    ans_tps = rec.get("avg_answer_tps")
                    tt = rec.get("avg_total_time_s")
                    msg = (
                        f"status={rec.get('status')}; "
                        f"acc={a if a is not None else 'NA'}; "
                        f"total_tps={tps if tps is not None else 'NA'}; "
                        f"answer_tps={ans_tps if ans_tps is not None else 'NA'}; "
                        f"total_time_s={tt if tt is not None else 'NA'}; "
                        f"param={param_name} value={v}; ok_n={rec.get('eval_ok_n')}"
                    )
                    notify(
                        "trial_done",
                        "trial done",
                        msg,
                        {
                            "trial_key": trial_key,
                            "trial_index": trial_index,
                            "status": rec.get("status"),
                            "avg_accuracy": a,
                            "avg_total_tps": tps,
                            "avg_total_time_s": tt,
                            "server_log": rec.get("server_log"),
                        },
                    )

            if rec.get("status") in ("failed", "partial"):
                notify("abnormal", "eval abnormal", f"param={param_name} value={v} status={rec.get('status')}", {
                    "trial_key": trial_key,
                    "eval_ok_n": rec.get("eval_ok_n"),
                    "server_log": rec.get("server_log"),
                })

            # 若达标且更快，立即通知
            if rec.get("status") == "ok" and rec.get("avg_accuracy") is not None and rec.get("avg_total_tps") is not None:
                try:
                    a = float(rec["avg_accuracy"])
                    tps = float(rec["avg_total_tps"])
                    if a >= float(args.accuracy_threshold) and tps > best_seen_total_tps:
                        best_seen_total_tps = tps
                        notify(
                            "best",
                            "new best",
                            f"acc={a:.4f}, total_tps={tps:.2f}, param={param_name} value={v}",
                            {
                                "trial_key": trial_key,
                                "avg_accuracy": a,
                                "avg_total_tps": tps,
                                "avg_total_time_s": rec.get("avg_total_time_s"),
                                "params": compact_params(trial.params, [
                                    "GPU_MEMORY_UTILIZATION",
                                    "VLLM_MAX_NUM_BATCHED_TOKENS",
                                    "VLLM_MAX_NUM_SEQS",
                                    "MAX_MODEL_LEN",
                                    "STOP_ON_DOUBLE_NEWLINE",
                                    "STOP_STRINGS",
                                ]),
                            },
                        )
                except Exception:
                    pass

            write_status(
                {
                    "phase": "trial_done",
                    "trial_index": trial_index,
                    "trial_key": trial_key,
                    "param": param_name,
                    "result": {
                        "status": rec.get("status"),
                        "avg_accuracy": rec.get("avg_accuracy"),
                        "avg_total_tps": rec.get("avg_total_tps"),
                        "avg_total_time_s": rec.get("avg_total_time_s"),
                    },
                    "best_params": base_params,
                }
            )
            time.sleep(args.cooldown)

        # 这一组（同一参数名）结束：从 group_results 里选最好并立刻写 best_params.json
        best_r = choose_best(group_results, acc_threshold=float(args.accuracy_threshold))
        if best_r is not None:
            base_params = dict(best_r.get("params") or base_params)
            atomic_write_json(best_path, base_params)
            write_status({"phase": "group_best", "param": param_name, "best_params": base_params})

    write_status({"phase": "done", "best_params": base_params, "results": os.path.relpath(results_path, repo)})
    notify("done", "tuning done", "auto_tune finished", {"best_params": base_params})
    print(f"Done. results={results_path} best={best_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        # 兜底：主循环外异常也尽量留下痕迹（无法保证在 SIGKILL 等情况下执行）
        try:
            p = repo_root_from_script()
            status_path = os.path.join(p, "tune_status.json")
            atomic_write_json(
                status_path,
                {
                    "ts": now_iso(),
                    "phase": "crashed",
                    "error": f"{type(e).__name__}: {e}",
                },
            )
        except Exception:
            pass
        raise
