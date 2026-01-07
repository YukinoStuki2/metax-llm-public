# Qwen2.5-0.5B Plus 推理服务

本项目提供了 Qwen2.5-0.5B Plus 大模型的高性能推理服务,支持 vLLM 加速和 Web 界面。

## 🚀 快速开始

### 使用 WebUI (推荐)

1. **启动推理后端**:
```bash
./run_model.sh
```

2. **启动 Web 界面**:
```bash
./start_webui.sh
```

3. 浏览器访问: http://localhost:7860

📖 详细文档请查看 [README_WEBUI.md](README_WEBUI.md)

---
## 融合 Adapter（LoRA/PEFT）到基座模型

如果你的微调仓库里只有 `adapter_model.safetensors`（没有完整的 merged 权重），可以在构建阶段下载基座模型并进行融合。

本项目默认在 Docker 构建阶段执行 [merge_adapter.py](merge_adapter.py)：

- 从 ModelScope 下载基座模型（默认 `Qwen/Qwen2.5-0.5B`）
- 从 Gitee clone 你的 adapter 仓库（默认 `https://gitee.com/yukinostuki/qwen2.5-0.5b-plus.git`）
- 使用 PEFT 将 adapter 融合到基座模型并导出到 `/app/model/merged`
- 运行时 `MODEL_DIR=/app/model/merged`

### 环境变量

- `ADAPTER_REPO_URL`：adapter 仓库地址（可用 https 或 ssh 地址）
- `ADAPTER_REPO_REF`：可选，指定分支/Tag/Commit
- `BASE_MODEL`：ModelScope 基座模型 ID，默认 `Qwen/Qwen2.5-0.5B`
- `BASE_REVISION`：基座模型 revision，默认 `master`
- `MERGED_MODEL_DIR`：融合输出目录，默认 `/app/model/merged`

### adapter_config.json 的要求

PEFT 融合需要 `adapter_config.json`。

如果你的 adapter 仓库里没有该文件（只提供了 `adapter_model.safetensors`），请额外提供其配置：

- `ADAPTER_CONFIG_JSON`：直接传 JSON 字符串
- 或 `ADAPTER_CONFIG_PATH`：指向一个 json 文件路径（构建阶段可用 COPY 注入）

否则融合脚本会报错并提示你补齐配置。

### WSL/本地运行（创建 Python 虚拟环境）

在 WSL（Ubuntu 24.04 等）里，如果你发现 `python3 -m venv` 报 `ensurepip is not available`，说明系统没装 venv/pip 组件。

1) 安装系统依赖（需要输入 sudo 密码）：

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip
```

2) 创建并激活虚拟环境：

```bash
cd /path/to/metax-demo-mirror
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

3) 安装 merge 所需 Python 依赖（推荐用最小集合）：

```bash
pip install -r requirements-merge.txt
```

4) 安装 PyTorch

- CPU-only（WSL/无 GPU 最稳）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

5) 运行融合：

```bash
python merge_adapter.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --adapter_repo_url https://gitee.com/yukinostuki/qwen2.5-0.5b-plus.git \
  --output_dir ./merged
```

如果你的 adapter 仓库里缺 `adapter_config.json`，按上文用 `ADAPTER_CONFIG_JSON` 或 `ADAPTER_CONFIG_PATH` 提供即可。

# 大模型推理服务模板(MetaX 沐曦)

本项目是一个极简的大模型推理服务模板，旨在帮助您快速构建一个可以通过API调用的推理服务器。

## 默认模型（ModelScope）

当前仓库默认直接从 ModelScope 下载模型权重：`YukinoStuki/Qwen2.5-0.5B-Plus-LLM`。

- 构建阶段由 `download_model.py` 下载到 `./model/$MODEL_ID`
- 运行阶段默认从 `MODEL_DIR=./model/$MODEL_ID` 加载（见 `Dockerfile` / `serve.py`）

默认 revision 为 `master`（可通过 `MODEL_REVISION` 覆盖，例如 tag/分支名）。

## 项目结构

- `Dockerfile`: 用于构建容器镜像的配置文件，MetaX提供docker构建流程，该文件中
    - `FROM`是通过拉取服务器中已存在的docker，因此不要进行改动
    - `EXPOSE`的端口是作为评测的主要接口，因此不要随意变动
- `serve.py`: 推理服务的核心代码，您需要在此文件中修改和优化您的模型加载与推理逻辑
    - `model_local_dict`: 是将模型映射到本地模型的dict
    - Notes: 这个程序不能访问Internet。
- `requirements.txt`: Python依赖列表。您可以添加需要的库。
- `.gitignore`: Git版本控制忽略的文件列表。
- `download_model.py`: 下载权重的脚本，不建议进行修改，模型参数、下载位置和版本都可以在`Dockerfile`中进行调整
    - `--model_name`: 模型名称，该名称是在modelscope中可以被检索的，例如：需要下载`DeepSeek-V3.2`,在modelscope中可知，`model_name`为`deepseek-ai/DeepSeek-V3.2`, 那么配置为`deepseek-ai/DeepSeek-V3.2`即可从modelscope中下载模型，如果您需要下载自己的微调模型，可以在modelscope中上传自己的模型，并调整该参数即可使用；
    - `--cache_dir`: 模型缓存地址，该地址是model存储的位置，例如指定下载`DeepSeek-V3.2`，在`/app`路径中，那么模型存放的位置在`/app/deepseek-ai/DeepSeek-V3.2`中；
    - `--revision`: 模型参数的Git版本，该版本对应modelscope仓库中的版本，您可根据自己微调数个版本，上传到同一仓库中，拉取时采用不同版本的revision即可；
    - Notes: 如果您的模型为非公开，请打开`download_model.py`进行相应的配置，本模板已将该部分注释([代码](download_model.py#L16-L17))，对注释内容取消注释并注入相应的内容即可配置非公开模型。在使用非公开模型时，建议在非judge环境中进行download_model的环境验证，以免浪费judge次数。
- `README.md`: 本说明文档

## 如何修改

您需要关注的核心文件是 `serve.py`.

您可以完全替换`serve.py`的内容，只要保证容器运行后，能提供模板中的'/predict'和'/'等端点即可。

## 评测系统的规则

评测系统会向 /predict 端点发送 POST 请求，其JSON body格式为: 
```json
{
  "prompt": "Your question here"
}
```
您的服务必须能够正确处理此请求，并返回一个JSON格式的响应，格式为: 
```json
{
  "response": "Your model's answer here"
}
```
请务必保持此API契约不变！

## Batch 推理（推荐提速）

评测系统在访问健康检查 `GET /` 时，如果返回 `{"status":"batch"}`，会进入 batch 模式：随后会把所有问题一次性推送到 `POST /predict`。

本仓库默认开启 batch（见 `Dockerfile` 的 `BATCH_MODE=1`）：

- 单条模式：`{"prompt": "..."}` → `{"response": "..."}`
- Batch 模式：`{"prompt": ["...", "...", ...]}` → `{"response": ["...", "...", ...]}`

本地评测脚本支持 batch 调用：

```bash
python eval_local.py --batch --strip_q_suffix --which bonus --max_n 50
```

## 自动化调参（auto_tune.py）

用于在本机反复“启动服务 → 跑评测 → 取均值 → 记录 → 关服”，自动搜索更快的运行参数。

说明：`auto_tune.py` 是严格串行的。每个 trial 都会按顺序执行：

1) 启动服务并等待健康检查通过
2) 跑 `eval_local.py` N 次并计算均值
3) 关闭服务并等待端口释放
4) 把本轮结果写入 `tune_results.jsonl`

只有完成（或明确失败并记录）后才会进入下一轮。

- 优化目标：在 `Accuracy >= 阈值` 前提下，最大化 `Throughput RAW: (prompt+answer)_tokens/s`。
- 结果输出：
  - `tune_results.jsonl`：每个试验一行（含均值、日志路径、失败原因等）
  - `best_params.json`：实时保存当前最优参数（后续试验会基于它继续搜索）
  - `tune_status.json`：实时状态（当前测试参数/进度），便于外部轮询查看

运行示例：

```bash
cd /data/metax-demo-mirror
source ./env_force.sh
./auto_tune.sh
```

### 容器环境保活（无 systemd / 无 systemctl）

很多云平台是在容器内运行（PID 1 不是 systemd），此时 `systemctl` 会报：

> System has not been booted with systemd as init system...

请使用本仓库提供的 watchdog：服务异常退出就自动重启；同时 `auto_tune.py` 会基于 `tune_results.jsonl` 自动断点续跑。

1) 更新代码并准备私密配置：

```bash
cd /data/metax-demo-mirror
git pull

cp tune_secrets.example.sh tune_secrets.sh
vim tune_secrets.sh
chmod 600 tune_secrets.sh
```

2) 可选：无模型自检（不启动模型，只校验解析/落盘/通知逻辑）：

```bash
./auto_tune.py --selftest --repo .
```

3) 后台启动 watchdog（断线不掉）：

```bash
nohup ./run_autotune_forever.sh > autotune.watchdog.log 2>&1 &
```

如需每 10 分钟发一次“我还活着”的飞书/邮件心跳（含已运行时长），在运行前设置：

```bash
export TUNE_HEARTBEAT_INTERVAL_S=600
# 可选：关闭“按 trial 次数心跳”（默认每 10 个 trial 一次）
export HEARTBEAT_TRIALS=0
```

4) 观察与停止：

```bash
# 看进程
ps -ef | grep -E 'run_autotune_forever|auto_tune.py' | grep -v grep

# 看日志
tail -f autotune.watchdog.log

# 看实时状态
cat tune_status.json

# 停止（pid 文件是“文本文件”，用 cat，不要当脚本执行）
cat autotune.watchdog.pid
kill "$(cat autotune.watchdog.pid)"
```

提示：如果容器本身被平台重建/重启，nohup 也无法存活；需要在平台侧设置“失败自动重启/始终运行”，并把启动命令设为 `./run_autotune_forever.sh`。

### 扩展探索组合（推荐：外部 search_space JSON）

默认搜索空间是有限的，跑完就会 `done` 退出。如果你希望在“合理范围内探索更多组合”，推荐用外部 JSON 扩展候选值：

1) 新建一个文件，例如 `tune_search_space.json`：

```json
{
  "GPU_MEMORY_UTILIZATION": ["0.965", "0.970", "0.975", "0.980", "0.985"],
  "VLLM_MAX_NUM_SEQS": ["1024", "1280", "1536"],
  "VLLM_MAX_NUM_BATCHED_TOKENS": ["131072", "196608", "262144"],
  "MAX_MODEL_LEN": ["1024", "1536", "2048"]
}
```

2) 指定路径并运行（会与默认 search_space 合并；新参数名会自动追加到探索顺序末尾）：

```bash
export TUNE_SEARCH_SPACE_FILE=./tune_search_space.json
./auto_tune.sh
```

如果你想“重新跑一遍”而不是跳过已跑组合，可以：
- 换一个结果文件：`RESULTS=tune_results_run2.jsonl ./auto_tune.sh`（或直接备份/清空旧文件）
- 或去掉 `--skip_existing`（不推荐在长任务里误删断点功能）

### 端口 8000 占用（port still in use）

若你看到飞书提示 `port 8000 still in use before start`，通常是容器里有残留服务占用 8000。

现在脚本会自动重试等待释放（默认重试 3 次、每次等 10 秒）。如果你确定占用者就是残留的 `uvicorn serve:app`，可以谨慎开启自动清理：

```bash
export TUNE_PORT_BUSY_RETRIES=6
export TUNE_PORT_BUSY_WAIT_S=10
export TUNE_PORT_BUSY_KILL=1
./auto_tune.sh
```

### 每轮结果通知（可选，发到飞书）

默认只在 `best/abnormal/done/start` 等关键事件通知。若你希望“每一轮跑完都看到本轮准确率/速度/耗时”，可以开启 per-trial 通知（注意：会比较多）：

```bash
export TUNE_NOTIFY_TRIAL_DONE=1
# 每 N 个 trial 发一次（1=每轮都发；例如 5=每 5 轮发一次）
export TUNE_NOTIFY_TRIAL_DONE_EVERY=1
./auto_tune.sh
```

### systemd 保活（仅适用于裸机/VM，有 systemd 的环境）

如果你的机器支持 systemd，可以参考示例服务文件 [autotune.service.example](autotune.service.example)。

```bash
sudo cp autotune.service.example /etc/systemd/system/autotune.service
sudo systemctl daemon-reload
sudo systemctl enable --now autotune.service

# 日志
sudo journalctl -u autotune.service -f
```

### 通知（可选）

1) Webhook（推荐，通用）：

```bash
export TUNE_WEBHOOK_URL="https://<your-webhook-endpoint>"
./auto_tune.sh
```

#### 飞书（推荐）

使用「飞书群自定义机器人」最省事：

- 在飞书群里：`群设置` → `机器人` → `添加机器人` → `自定义机器人`
- 复制 `Webhook 地址`
- （可选）在机器人「安全设置」里开启 `签名校验`，得到 `secret`

然后在评测机上设置：

```bash
export TUNE_WEBHOOK_KIND=feishu
export TUNE_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
# 可选：如果你开启了签名校验
export TUNE_FEISHU_SECRET="your_secret"

./auto_tune.sh
```

2) SMTP 邮件：

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

# 可选：控制发信类型（best/crashed/done/abnormal）
# export TUNE_EMAIL_KINDS="best,crashed,done"

./auto_tune.sh
```

#### 163 邮箱兜底（示例）

163 通常需要在邮箱设置里开启 SMTP，并生成“授权码”（不是登录密码）。

```bash
export TUNE_SMTP_HOST="smtp.163.com"
export TUNE_SMTP_PORT="465"
export TUNE_SMTP_USER="imuner@163.com"
export TUNE_SMTP_PASS="<163授权码>"
export TUNE_SMTP_FROM="imuner@163.com"
export TUNE_SMTP_TO="imuner@163.com"
export TUNE_SMTP_SSL=1

# 邮件默认只发 best/crashed/done；如需更多：
# export TUNE_EMAIL_KINDS="best,crashed,done,abnormal"

./auto_tune.sh
```

## Speculative Decoding（冲吞吐，可选）

本仓库已在 [serve.py](serve.py) 接入 vLLM 的 speculative decoding（需要一个 draft 小模型）。

- 启用方式（运行期）：
  - `ENABLE_SPECULATIVE_DECODING=1`
  - 两种模式：
    - draft 模型（推荐）：指定 `SPEC_DRAFT_MODEL_DIR=/app/model/<draft>`（或用 `SPEC_DRAFT_MODEL_ID` 让构建期下载），并保持 `SPEC_METHOD=draft_model`
    - ngram（无需 draft，快速试验）：`SPEC_METHOD=ngram`，可调 `SPEC_NGRAM_LOOKUP_MAX=8`/`SPEC_NGRAM_LOOKUP_MIN=1`
  - 可调：`SPEC_NUM_SPECULATIVE_TOKENS=6`（建议 4~8）

- 构建期下载 draft（Dockerfile 已透传参数，默认 draft 下载失败不阻断构建）：
  - `SPEC_DRAFT_MODEL_ID=<ModelScope 模型 ID>`
  - `SPEC_DRAFT_MODEL_REVISION=master`

注意：vLLM 当前实现中 speculative decoding 与 chunked prefill 不兼容；当启用 speculative 时，服务端会强制关闭 `enable_chunked_prefill`。

## AWQ 量化（AutoAWQ，覆盖上传同名模型）

说明：部分环境/架构不支持 Marlin kernel，因此此前 compressed-tensors 路线可能无法运行。这里提供 AutoAWQ 量化脚本，输出目录固定为 `model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ`，可直接用上传脚本覆盖同名仓库。

1) 安装量化依赖（建议单独虚拟环境；不修改线上 serving 的 requirements.txt）：

```bash
python -m venv .venv-awq
source .venv-awq/bin/activate

pip install -U pip setuptools wheel

# 先安装 transformers 等轻依赖（避免 autoawq 自动升级 torch 破坏环境）
pip install -r requirements-quantize-awq.txt

# 再安装 autoawq（不自动拉取/升级 torch）
pip install --no-deps autoawq==0.2.9
```

2) 量化并导出到固定目录：

```bash
python quantize_awq_llmcompressor.py \
  --model_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM \
  --calib_jsonl calib_512.jsonl \
  --output_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ
```

3) 上传并覆盖 ModelScope 上同名仓库：

```bash
python upload_model.py \
  --repo-id YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ \
  --model-dir model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ \
  --commit-message "overwrite awq (autoawq)"
```

## 环境说明

### 软件包版本

主要软件包(vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64)版本如下：

|软件|版本|
|:--:|:--:|
|python|3.10|
|ubuntu|22.04|
|pytorch|2.6|
|vLLM|0.10.0|

`软件使用的Note`:
- 如果您需要其他的镜像，请您先查询[沐曦开发者社区](https://developer.metax-tech.com/softnova/docker)，查找您需要的docker镜像，后联系龚昊助教添加相应的镜像。
- 建议您先在`OpenHydra`中使用添加的软件，避免软件兼容性带来的问题（非GPU相关的软件都可以兼容，GPU相关软件或依赖GPU相关软件的软件建议验证后使用）。
- `OpenHydra`的访问地址请查询`沐曦GPU实验平台操作手册`，欢迎您的使用。

### judge平台的配置说明

judge机器的配置如下：

``` text
os: ubuntu24.04
cpu: 24核
内存: 200GB
磁盘: 1T
GPU: MXC500(显存：64GB)
网络带宽：100Mbps
```

judge系统的配置如下：

``` text
docker build stage: 900s
docker run - health check stage: 180s
docker run - predict stage: 360s
```
