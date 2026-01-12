---
title: 自动调参（auto_tune.py / auto_tune.sh）
sidebar_position: 50
---

本仓库提供一套“守护式自动调参”脚本：对一组候选环境变量组合反复执行：

1. 启动推理服务（调用 `run_model.sh`）

2. 轮询健康检查（访问 `GET /`）

3. 跑 N 次本地评测（调用 `eval_local.py`）

4. 解析输出得到准确率/吞吐/耗时，并写入结果文件

5. 关闭服务（杀进程组 + 等端口释放）

6. 进入下一组参数

相关脚本分工：

- `auto_tune.sh`：入口脚本（会自动 `source ./tune_secrets.sh`，若存在），把常用环境变量映射为 `auto_tune.py` 参数

- `auto_tune.py`：核心逻辑（仅用标准库），负责：搜索空间、启动/停服、跑评测、解析、落盘、通知

- `run_autotune_forever.sh`：无 root 的 watchdog，自恢复重启（`auto_tune.sh` 异常退出就重启）

- `tune_secrets.example.sh`：通知/私密配置示例（复制为 `tune_secrets.sh`）

- `autotune.service.example`：可选 systemd 服务示例（需要 systemd 环境）

---

## 1) 脚本功能概述

### 1.1 适用场景

- 已能跑通服务与评测，希望在固定评测口径下自动寻找“准确率达标且吞吐更高”的一组推理参数。

- 需要长时间连续运行：单轮启动失败/评测超时/服务崩溃会被记录，随后继续探索下一组。

### 1.2 一次调参试验（trial）会做什么

对某组参数 `trial.params`，`auto_tune.py` 会：

1. 检查端口（默认 8000）是否空闲；必要时等待/重试；可选只杀“明显是残留的 `uvicorn serve:app`”进程

2. 以该参数组作为环境变量启动 `bash run_model.sh`（单独进程组，日志写入 `tune_server_logs/trial_XXXXX.log`）

3. 轮询等待服务就绪（`GET /` 200\~499 视为可用）

4. 串行运行 `eval_local.py` N 次（默认 5 次），对每次输出解析：

   - `Accuracy (RougeL-F1 mean, RAW): ...`

   - `Throughput RAW: ... (prompt+answer)_tokens/s=...`

   - `Total time: ...s`

5. 杀掉服务进程组并等待端口释放

6. 写结果 JSONL，并更新 `tune_status.json`

### 1.3 快速开始（推荐流程）

建议先把环境变量清干净（避免上一次调参遗留影响本轮）：

```bash
source ./env_force.sh
```

如需通知功能，先准备 `tune_secrets.sh`（可选）：

```bash
cp tune_secrets.example.sh tune_secrets.sh
chmod 600 tune_secrets.sh
```

启动自动调参：

```bash
./auto_tune.sh
```

常用覆盖（门槛 + 重复次数）：

```bash
ACC=0.8800 EVAL_RUNS=5 ./auto_tune.sh
```

可选：无模型自检（不启动模型，只校验解析/落盘/断点/选优逻辑）：

```bash
./auto_tune.py --selftest --repo .
```

### 1.4 结果输出（文件与字段）

默认输出都位于 repo 根目录：

- `tune_results.jsonl`

  - 每行一条记录（JSON），核心字段：

    - `trial_key`：参数集合的稳定 key（按参数名排序拼接，方便断点续跑）

    - `status`：`ok | partial | failed | timeout | crashed`

    - `phase`：`startup` 或 `eval`

    - `params`：本轮传给服务的环境变量（字符串）

    - `eval_runs`：每次评测的明细列表（包含 accuracy/throughput/time/elapsed/returncode 等）

    - `avg_accuracy / avg_total_tps / avg_answer_tps / avg_total_time_s`：均值（仅对成功 run 聚合）

    - `server_log`：本轮服务日志路径（相对 repo）

- `best_params.json`

  - “当前最优参数”（只保存 params，不保存分数）

  - 更新时机：每个参数组（同一 param_name）探索结束后，选出达标且 `avg_total_tps` 最高的一条，立刻覆盖写入

- `tune_status.json`

  - 实时状态（json）：用于外部监控/看当前跑到哪一轮、最近结果、当前 base_params

- `tune_server_logs/`

  - 每轮启动服务的原始 stdout/stderr（合并输出）

### 1.5 守护运行（可选）

无 root watchdog（最通用）：

```bash
./run_autotune_forever.sh
```

- 它会写一个 `autotune.watchdog.pid`，便于定位 watchdog PID。

systemd（只在有 systemd 的机器上）：参考 `autotune.service.example`。

### 1.6 容器环境后台运行与观察

当运行环境不是 systemd（例如容器内 PID 1 不是 systemd）时，`systemctl` 可能会报错：

```text
System has not been booted with systemd as init system...
```

此时可用仓库内置的 watchdog 进行保活：

```bash
nohup ./run_autotune_forever.sh > autotune.watchdog.log 2>&1 &
```

心跳（可选）：按时间间隔发一次“仍在运行”的通知。

```bash
export TUNE_HEARTBEAT_INTERVAL_S=600
# 可选：关闭“按 trial 次数心跳”（默认每 10 个 trial 一次）
export HEARTBEAT_TRIALS=0
```

观察与停止：

```bash
# 看进程
ps -ef | grep -E 'run_autotune_forever|auto_tune.py' | grep -v grep

# 看 watchdog 日志
tail -f autotune.watchdog.log

# 看实时状态
cat tune_status.json

# 停止（pid 文件是文本文件）
cat autotune.watchdog.pid
kill "$(cat autotune.watchdog.pid)"
```

说明：如果容器被平台重建/重启，nohup 也无法跨重启存活；需要在平台侧设置“失败自动重启/始终运行”，并把启动命令设为 `./run_autotune_forever.sh`。

---

## 2) 参数与环境变量详解

这一节分两层：

1. **“调参脚本自己的参数/变量”**（影响调参流程、通知、端口占用策略等）

2. **“被调参的服务参数”**（实际会作为环境变量注入到 `run_model.sh`，影响 `serve.py` 行为）

### 2.1 `auto_tune.sh` 入口环境变量（最常用）

`auto_tune.sh` 会把下面这些变量转换为 `auto_tune.py` 的命令行参数：

- `REPO`：仓库路径，默认 `.`

- `EVAL_RUNS`：每个 trial 跑几次评测，默认 `5`

- `ACC`：准确率门槛（`avg_accuracy >= ACC` 才可能被认为是“可用最优”），默认 `0.8800`

- `STARTUP_TIMEOUT`：等待服务启动的秒数，默认 `240`

- `EVAL_TIMEOUT`：单次 `eval_local.py` 的超时秒数，默认 `420`

- `HEARTBEAT_TRIALS`：每 N 个 trial 发一次心跳通知，默认 `10`

- `HEARTBEAT_INTERVAL_S`：每隔 N 秒发一次心跳通知（0=关闭）；默认读取 `TUNE_HEARTBEAT_INTERVAL_S`，否则 0

- `SEARCH_SPACE_FILE`：外部搜索空间 JSON 文件路径（可选）；默认读取 `TUNE_SEARCH_SPACE_FILE`

- `PORT_BUSY_RETRIES`：端口占用重试次数，默认读取 `TUNE_PORT_BUSY_RETRIES`，否则 3

- `PORT_BUSY_WAIT_S`：端口占用每次等待秒数，默认读取 `TUNE_PORT_BUSY_WAIT_S`，否则 10

- `PORT_BUSY_KILL`：端口占用时是否尝试清理残留 `uvicorn serve:app`（谨慎），默认读取 `TUNE_PORT_BUSY_KILL`，否则 0

- `NOTIFY_TRIAL_DONE`：每轮 trial 都通知（会很多），默认读取 `TUNE_NOTIFY_TRIAL_DONE`，否则 0

- `NOTIFY_TRIAL_DONE_EVERY`：每 N 个 trial 发一次 trial_done 通知，默认读取 `TUNE_NOTIFY_TRIAL_DONE_EVERY`，否则 1

可直接用一行覆盖：

```bash
ACC=0.8810 EVAL_RUNS=3 STARTUP_TIMEOUT=300 ./auto_tune.sh
```

### 2.2 `auto_tune.py` 命令行参数（完整）

`auto_tune.py` 的关键参数与默认值：

- 运行基础

  - `--repo`：仓库路径（默认脚本所在目录）

  - `--port`：服务端口（默认 8000）

  - `--eval_runs`：每轮评测次数（默认 5）

  - `--startup_timeout`：等服务 ready 的秒数（默认 240）

  - `--eval_timeout`：单次评测超时秒数（默认 420）

  - `--cooldown`：每轮 trial 结束后 sleep 秒数（默认 1.0）

  - `--accuracy_threshold`：选择最优参数时的准确率门槛（默认 0.8800）

- 输出路径（相对 `--repo`）

  - `--results`：结果 JSONL（默认 `tune_results.jsonl`）

  - `--best`：最优参数 JSON（默认 `best_params.json`）

  - `--server_log_dir`：服务日志目录（默认 `tune_server_logs`）

  - `--status_file`：状态 JSON（默认 `tune_status.json`）

- 心跳与断点

  - `--heartbeat_trials`：每 N 个 trial 发心跳（默认 10；0=关闭）

  - `--heartbeat_interval_s`：每隔 N 秒发心跳（默认来自 `TUNE_HEARTBEAT_INTERVAL_S`；0=关闭）

  - `--skip_existing`：跳过 results 中已完成的 trial（`ok/failed/crashed/timeout`）

- 搜索空间

  - `--search_space_file`：外部搜索空间 JSON（默认来自 `TUNE_SEARCH_SPACE_FILE`）

- 端口占用处理

  - `--port_busy_retries`：重试次数（默认来自 `TUNE_PORT_BUSY_RETRIES` 或 3）

  - `--port_busy_wait_s`：每次等待秒数（默认来自 `TUNE_PORT_BUSY_WAIT_S` 或 10）

  - `--port_busy_kill`：谨慎开关，尝试只杀残留的 `uvicorn serve:app`

- 通知（可选）

  - `--notify_trial_done`：每轮都通知（默认来自 `TUNE_NOTIFY_TRIAL_DONE`）

  - `--notify_trial_done_every`：每 N 轮通知一次（默认来自 `TUNE_NOTIFY_TRIAL_DONE_EVERY` 或 1）

  - `--smtp_host/--smtp_port/--smtp_user/--smtp_pass/--smtp_from/--smtp_to`：SMTP 配置

  - `--smtp_no_starttls`：关闭 STARTTLS

  - `--smtp_ssl`：使用 SMTPS（例如 465）

  - `--email_kinds`：哪些事件发邮件，默认来自 `TUNE_EMAIL_KINDS`（默认 `best,crashed,done`）

  - `--webhook_url`：webhook URL（默认来自 `TUNE_WEBHOOK_URL`）

  - `--webhook_kind`：`feishu | generic`（默认来自 `TUNE_WEBHOOK_KIND`，默认 `feishu`）

  - `--feishu_secret`：飞书签名 secret（默认来自 `TUNE_FEISHU_SECRET`）

- 调试/自检

  - `--dry_run`：只生成 trial 列表估算数量并退出（不启动模型）

  - `--selftest`：不启动模型的自检（解析/落盘/断点/选优逻辑）

  - `--selftest_notify`：selftest 时也尝试发通知（需要已配置）

### 2.3 `tune_secrets.sh`（通知与可选开关）

脚本会自动加载 `./tune_secrets.sh`（若存在）；通知/密钥/个性化开关通常放在该文件中统一管理。

推荐做法：

```bash
cp tune_secrets.example.sh tune_secrets.sh
chmod 600 tune_secrets.sh
```

常用变量：

- 飞书：

  - `TUNE_WEBHOOK_KIND=feishu`

  - `TUNE_WEBHOOK_URL`：机器人 webhook URL

  - `TUNE_FEISHU_SECRET`：签名 secret（不确定就留空，避免签名失败）

- 邮件（兜底）：

  - `TUNE_SMTP_HOST / TUNE_SMTP_PORT / TUNE_SMTP_SSL / TUNE_SMTP_NO_STARTTLS`

  - `TUNE_SMTP_USER / TUNE_SMTP_PASS / TUNE_SMTP_TO / TUNE_SMTP_FROM`

  - `TUNE_EMAIL_KINDS`：默认 `best,crashed,done`

### 2.4 被调参的“服务环境变量”（`auto_tune.py` 会注入到 `run_model.sh`）

`auto_tune.py` 会把当前 trial 的 `params` 写入子进程环境变量，然后启动 `bash run_model.sh`。

两类来源：

1. **base_params**：来自 `best_params.json`（若存在），否则使用脚本内置兜底默认（与 `env_force.sh` 语义对齐）

2. **search_space**：脚本内置搜索空间 + 可选外部 JSON 合并；按 `order` 指定的顺序逐组探索

内置兜底默认（当 `best_params.json` 不存在或为空时使用）包括：

- 生成长度与分流：`MAX_NEW_TOKENS=64`、`MAX_NEW_TOKENS_CODE=192`、`MAX_NEW_TOKENS_CODE_SOFT=96`、`MAX_NEW_TOKENS_CODE_HARD=192`

- 长答/代码题启发式：`HARD_CODE_MIN_HITS=1`、`LONG_ANSWER_ENABLE_DEFAULT=1`、`LONG_ANSWER_MIN_HITS=1`

- 解码稳定性：`REPETITION_PENALTY=1.05`、`FREQUENCY_PENALTY=0.1`

- stop：`STOP_STRINGS=<|im_end|>,<|endoftext|>`、`STOP_ON_DOUBLE_NEWLINE=0`

- vLLM 容量：`GPU_MEMORY_UTILIZATION=0.97`、`VLLM_MAX_NUM_BATCHED_TOKENS=131072`、`VLLM_MAX_NUM_SEQS=1024`、`MAX_MODEL_LEN=1024`

- 输出后处理：`OUTPUT_TRIM_EXAMPLES=1`、`OUTPUT_MAX_SENTENCES=6`

内置搜索空间（节选）包括：

- vLLM 相关：`GPU_MEMORY_UTILIZATION`、`VLLM_MAX_NUM_BATCHED_TOKENS`、`VLLM_MAX_NUM_SEQS`、`MAX_MODEL_LEN`

- stop 与裁剪：`STOP_ON_DOUBLE_NEWLINE`、`STOP_STRINGS`、`OUTPUT_TRIM_EXAMPLES`、`OUTPUT_MAX_SENTENCES`

- 生成长度：`MAX_NEW_TOKENS`、`MAX_NEW_TOKENS_CODE*`

- 惩罚项：`REPETITION_PENALTY`、`FREQUENCY_PENALTY`

探索顺序（order）是“先稳定再榨吞吐再微调输出”的坐标上升式流程：

1. 先找到更稳的显存利用率

2. 再挤压吞吐（batch tokens、并发序列、max_model_len）

3. 再调 stop 与输出裁剪

4. 再调长度分流

5. 最后才动惩罚项（更可能影响准确率）

### 2.5 调参加速相关变量（由 `auto_tune.py` 默认注入）

为避免每轮 trial 都重复安装/下载，`auto_tune.py` 在启动 `run_model.sh` 时会：

- `env.setdefault("SKIP_PIP_INSTALL", "1")`

- `env.setdefault("SKIP_MODEL_DOWNLOAD", "1")`

含义：

- 未显式设置这两个变量时，脚本默认开启“跳过 pip 安装/跳过模型下载”。

- 显式设置 `SKIP_PIP_INSTALL=0`（或其他值）时，脚本会尊重该设置。

---

## 3) 代码详解（实现逻辑与关键函数）

### 3.1 解析评测输出：`parse_eval_output`

`auto_tune.py` 不直接依赖 `eval_local.py` 的内部数据结构，而是用正则从 stdout 抽取：

- `Accuracy (RougeL-F1 mean, RAW): ...`

- `Throughput RAW: ... answer_tokens/s=... (prompt+answer)_tokens/s=...`

- `Total time: ...s`

这让调参脚本对评测输出格式非常敏感：一旦 `eval_local.py` 的 summary 文案改了，解析就会失败（见 FAQ）。

### 3.2 端口与健康检查

- `is_port_open`：用 socket 连接探测端口。

- `http_get_localhost`：最小 HTTP GET（不引入 requests），用于访问 `GET /`。

- `wait_server_ready`：

  - 先等端口开放

  - 再请求 `GET /`

  - 只要返回 HTTP 状态在 200\~499 就视作“服务活着”（不要求一定是 `{"status":"ok"}`），以兼容 batch/ready 状态。

### 3.3 端口占用清理（可选）：`try_free_port`

默认策略是“等待 + 重试”。当启用 `--port_busy_kill`（或 `TUNE_PORT_BUSY_KILL=1`）时，会：

1. 用 `ss -ltnp` 尝试解析监听指定端口的 pid

2. 读取 `/proc/<pid>/cmdline`

3. 只对同时包含 `uvicorn` 和 `serve:app` 的进程发信号（SIGTERM，再 SIGKILL）

这是一种“尽量不误伤”的做法，但仍然建议谨慎开启。

### 3.4 通知系统：`notify`

事件类型（kind）在脚本里会出现：

- `start`：调参开始

- `heartbeat`：心跳

- `trial_done`：每轮结束（可选开关）

- `best`：发现新 best（达标且更快）

- `failed/timeout/crashed/abnormal`：启动/评测异常

- `done`：调参结束

通知渠道：

- webhook（优先）：

  - `feishu`：发送交互卡片，且飞书需要额外判断 JSON `code==0` 才算成功

  - `generic`：普通 JSON POST

- email（兜底）：只在 `kind in email_kinds` 时发送；默认只发 `best,crashed,done`

### 3.5 搜索算法：坐标上升（按参数分组）

脚本不是做“全组合笛卡尔积”，而是更节省试验数的坐标上升式：

1. 给定 `base_params`（来自 `best_params.json` 或内置默认）

2. 按 `order` 逐个参数名（param group）进行探索

3. 同一参数名下，会测试：`当前值` + `候选列表`（去重后）

4. 每个 trial 都会跑完整流程并写入 `tune_results.jsonl`

5. 一个参数组结束后：

   - 过滤 `avg_accuracy >= accuracy_threshold`

   - 按 `avg_total_tps`（prompt+answer tokens/s）降序选最优

   - 把该条的 `params` 覆盖写入 `best_params.json`，并作为新的 `base_params` 进入下一组

这样做的取舍：

- 优点：试验数量线性增长，适合长时间跑；且会实时落盘 best，随时可中断/续跑。

- 缺点：不能保证全局最优（局部最优风险），但通常足够找到“达标且更快”的组合。

### 3.6 子进程生命周期：启动/评测/关服

每轮会启动一个独立服务进程组：

- `subprocess.Popen(["bash", "run_model.sh"], start_new_session=True)`

- stdout/stderr 合并写入该轮日志文件

- 评测结束后无论成功与否都会 `kill_process_group(p)`

这保证了：

- `auto_tune.py` 崩了或被 SIGTERM，信号处理器会尽量杀掉正在运行的服务，减少端口残留。

### 3.7 评测命令是“固定写死”的

当前代码里 `eval_cmd` 是固定列表（不是参数化的）：

- `sys.executable eval_local.py --which bonus --model_dir_for_tokenizer ./model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM --batch --overwrite_jsonl`

含义：

- 调参使用的准确率/吞吐口径是固定的（bonus 子集、batch 模式、固定 tokenizer）。

- 如需切换 `--which` 或 tokenizer 目录，需要修改 `auto_tune.py` 的 `eval_cmd` 段。

---

## 4) 常见问题

### 4.1 为什么一直显示端口占用（port busy）？

原因：

- 上一轮服务进程没有被正确退出（例如机器卡死/手动中断时留下残留）。

处理：

- 先看 `tune_server_logs/trial_XXXXX.log` 的末尾是否有异常。

- 轻度处理：增大等待与重试：`TUNE_PORT_BUSY_RETRIES=5 TUNE_PORT_BUSY_WAIT_S=20`。

- 仅在确认占用者是残留的 `uvicorn serve:app` 时，再谨慎开启：`TUNE_PORT_BUSY_KILL=1`。

### 4.2 为什么 trial 大量是 timeout/crashed？

常见原因：

- `run_model.sh` 启动过慢（模型加载、vLLM 初始化、预热），超过 `STARTUP_TIMEOUT`。

- 某组参数导致 vLLM OOM 或启动失败。

处理：

- 先调大：`STARTUP_TIMEOUT=400`。

- 如果大量是 OOM：缩小搜索空间里 `GPU_MEMORY_UTILIZATION` 或 `MAX_MODEL_LEN` 的上限。

- 优先确保单次能稳定跑通：手动用 `env_force.sh && ./run_model.sh` 验证。

### 4.3 为什么评测解析不到准确率/吞吐？

原因：`auto_tune.py` 用正则解析 `eval_local.py` 的 stdout 固定文案。

处理：

- 检查 `eval_local.py` 是否仍输出类似：

  - `Accuracy (RougeL-F1 mean, RAW): ...`

  - `Throughput RAW: ... (prompt+answer)_tokens/s=...`

- 若调整了 `eval_local.py` 的 summary 文案，需要同步更新 `auto_tune.py` 的正则。

### 4.4 best_params.json 里为什么没有分数？

这是设计使然：

- `best_params.json` 只存“参数”，便于作为下一轮 base_params 或直接拿去部署。

- 分数与明细都在 `tune_results.jsonl` 里。

### 4.5 为什么我开启了通知，但没有收到飞书消息？

排查顺序：

- 确认 `auto_tune.sh` 确实加载了 `tune_secrets.sh`（文件存在且可读）。

- 确认 `TUNE_WEBHOOK_URL` 非空。

- 如果启用了飞书签名：

  - `TUNE_FEISHU_SECRET` 填错会导致常见错误（例如签名校验失败）。不确定就留空。

- 注意：飞书 webhook HTTP 200 不代表成功，脚本会进一步检查响应 JSON 的 `code`。

### 4.6 为什么邮件发不出去？

常见原因：

- `TUNE_SMTP_*` 配置不完整，或密码应填“授权码”而不是登录密码。

- 465 端口需要 SMTPS：设置 `TUNE_SMTP_SSL=1` 并把 `TUNE_SMTP_PORT=465`。

- 587 常用 STARTTLS：确保没有设置 `TUNE_SMTP_NO_STARTTLS=1`。

另外：默认只发 `best,crashed,done`，如需 `abnormal` 也发送，设置：

```bash
export TUNE_EMAIL_KINDS="best,crashed,done,abnormal"
```

### 4.7 为什么我希望调更多参数，但脚本没测？

- 内置 `order` 只覆盖一组常用参数。

- 可用外部 JSON 合并：`TUNE_SEARCH_SPACE_FILE=...`。

- 若增加新的参数名：脚本会把它追加到探索顺序最后（不会影响已有顺序）。

### 4.8 如何扩展探索组合（外部 search_space JSON）

内置搜索空间是有限的，跑完就会 `done` 退出。可用外部 JSON 扩展候选值：

1. 新建一个文件，例如 `tune_search_space.json`：

```json
{
  "GPU_MEMORY_UTILIZATION": ["0.965", "0.970", "0.975", "0.980", "0.985"],
  "VLLM_MAX_NUM_SEQS": ["1024", "1280", "1536"],
  "VLLM_MAX_NUM_BATCHED_TOKENS": ["131072", "196608", "262144"],
  "MAX_MODEL_LEN": ["1024", "1536", "2048"]
}
```

1. 指定路径并运行（会与默认 search_space 合并；新参数名会自动追加到探索顺序末尾）：

```bash
export TUNE_SEARCH_SPACE_FILE=./tune_search_space.json
./auto_tune.sh
```

重新跑一遍而不是跳过已跑组合，可选方式：

- 更换结果文件：`RESULTS=tune_results_run2.jsonl ./auto_tune.sh`

- 或关闭断点跳过（不建议在长任务里误删断点功能）：去掉 `--skip_existing`

### 4.9 如何开启“每轮结果通知”（trial_done）

默认只对 `best/abnormal/done/start` 等关键事件通知。需要每一轮都通知时：

```bash
export TUNE_NOTIFY_TRIAL_DONE=1
# 1=每轮都发；例如 5=每 5 轮发一次
export TUNE_NOTIFY_TRIAL_DONE_EVERY=1
./auto_tune.sh
```

### 4.10 如何配置飞书 webhook 通知

使用「飞书群自定义机器人」即可。

- 在飞书群里：群设置 → 机器人 → 添加机器人 → 自定义机器人

- 复制 webhook 地址

- 如开启“签名校验”，记录 secret

运行前设置：

```bash
export TUNE_WEBHOOK_KIND=feishu
export TUNE_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
# 可选：开启签名校验时配置
export TUNE_FEISHU_SECRET="your_secret"

./auto_tune.sh
```

### 4.11 如何配置 SMTP 邮件通知

示例：

```bash
export TUNE_SMTP_HOST="smtp.example.com"
export TUNE_SMTP_PORT="587"
export TUNE_SMTP_USER="user@example.com"
export TUNE_SMTP_PASS="your_password_or_token"
export TUNE_SMTP_FROM="user@example.com"
export TUNE_SMTP_TO="you@example.com"

# 587 常用 STARTTLS；如需禁用 STARTTLS：
# export TUNE_SMTP_NO_STARTTLS=1

# 465 常用 SMTPS（SSL）；如需启用：
# export TUNE_SMTP_SSL=1

./auto_tune.sh
```

### 4.13 我想换评测子集/换 tokenizer/换 eval 参数

当前 `auto_tune.py` 的 `eval_cmd` 是写死的。

如需切换：

- 修改 `auto_tune.py` 中 `eval_cmd = [...]` 段。

- 同时确认 `parse_eval_output` 的正则仍能解析到所需指标。