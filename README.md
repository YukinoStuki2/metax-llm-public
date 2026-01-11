# 微调Qwen大模型的推理部署与工具

**2025年秋季中国科学院大学(国科大)《GPU架构与编程》课程项目二**

📖 详细文档请查看：<https://docs-gpu.yukino.uk>

## 📌 导航

- 🚀 快速开始

- 使用说明

- 🧩 API 契约

- 🗂️ 文件/目录说明

---

## 摩尔线程一等奖——沐曦赛道

---

本项目提供了 vLLM 加速的推理代码和 Web 界面以及一些测试模型的工具脚本。

Gitee仓库（用于实际提交）：<https://gitee.com/yukinostuki/metax-demo>

Github仓库（用于具体开发）：<https://github.com/YukinoStuki2/metax-llm-public>

两个仓库仅有文件数量区别，其版本和代码内容都是同步的。

## 项目介绍

这是一个对微调开源大模型进行推理，用于问答评测的项目：

- 微调数据集来源于《Programming Massively Parallel Processors.2017》。

- 评测目标：在准确率（RougeL-F1，jieba 分词）达到阈值（常见参考 ≥ 0.35）的前提下，尽量提升吞吐（tokens/s）。

评测系统关键约束：

- Build 阶段允许联网（用于下载依赖/权重）；Run 阶段断网（请求路径内不做任何联网操作）。

- 必须提供：

  - `GET /`：健康检查（返回“ok”或“batch”切换模式）

  - `POST /predict`：推理接口（见下文 API 契约）

- 端口：保持 `8000`。

- 评测机配置：

  - OS：Ubuntu 24.04

  - CPU：24 核

  - 内存：200GB

  - 磁盘：1TB

  - GPU：MXC500（64GB 显存）

  - 网络：100Mbps

- 时间限制

  * docker build：900s

  * health（`GET /`）：180s

  * predict（推理阶段总计）：360s

## 🚀 快速开始

本项目提供了两种启动方式，可以使用Dockerfile进行docker启动，也可以直接通过bash启动。

**请保证你已经有如下前置环境：**

主要软件包(vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64)版本如下：

|软件|版本|
|-|-|
|python|3.10|
|ubuntu|22.04|
|pytorch|2.6|
|vLLM|0.10.0|

**测试时使用 Gitee 提供的沐曦算力服务器和系统镜像进行部署，实测没有问题；其他平台请注意前置依赖是否满足。**

### 使用 WebUI (推荐)

1. 导入环境变量：

```bash
source ./env_force.sh
```

1. 启动推理后端：

```bash
bash ./run_model.sh
```

1. 启动 Web 界面：

```bash
./start_webui.sh
```

1. 浏览器访问：<http://localhost:7860>

5.（可选）如果在沐曦服务器中启动，可以通过 SSH 隧道连接 WebUI：

```shell
ssh -CNg -L 7860:127.0.0.1:7860 root+<username>@<IP> -p <PORT>
```

### 使用Docker启动（评测系统无webui）

**！沐曦容器上无法直接用docker启动**

``` text
docker build stage: 900s
docker run - health check stage: 180s
docker run - predict stage: 360s
```

启动后自测健康检查：

```bash
curl -s http://127.0.0.1:8000/
```

单条推理请求：

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"请简要回答：什么是xxx？"}'
```

---

## 使用说明

---

## 🧩 API 契约

评测机将按照如下方式调用服务（**不要破坏**）：

- 端口：`8000`

- `GET /`：健康检查

- `POST /predict`：

  - 请求 JSON：`{"prompt":"..."}`

  - 响应 JSON：`{"response":"..."}`

本项目也支持 batch：

- 当 `BATCH_MODE=1` 时，`GET /` 会返回 `{"status":"batch"}`，评测系统会将所有问题一次性推到 `POST /predict`。

- batch 请求格式：`{"prompt":["...","...",...]}`

- batch 响应格式：`{"response":["...","...",...]}`（答案数量需与问题数量一致）

---

## 🗂️ 文件/目录说明

### 根目录（核心脚本与配置）

|路径|用途|
|-|-|
|`.dockerignore`|控制 Docker build 时上下文包含/排除哪些文件，减少体积与构建时间。|
|`.gitignore`|Git 忽略规则（本地模型、日志、虚拟环境、调参产物等）。|
|`Dockerfile`|评测用镜像构建入口：安装依赖、下载/准备模型，并以 `serve.py` 提供 HTTP 服务。|
|`DEPLOY.md`|部署相关说明（如环境、启动方式、注意事项）。|
|`QUICKREF.md`|常用命令速查（本地评测、启动、调参等）。|
|`README.md`|GitHub 主 README（你正在阅读的这份）。|
|`README_GITEE.md`|Gitee 提交/评测用精简 README（同步工作流会用它覆盖到 Gitee 的 README）。|
|`README_WEBUI.md`|WebUI 使用说明与常见问题。|
|`env_force.sh`|**强制导入**一整套默认环境变量（用于清理上次遗留变量；需用 `source` 执行）。|
|`run_model.sh`|启动推理服务（本地/云主机复现评测流程用；读取本地变量执行，不强行覆盖外部变量）。|
|`serve.py`|FastAPI/uvicorn 服务端实现：`GET /` + `POST /predict`。|
|`download_model.py`|构建阶段下载模型权重到指定目录。|
|`webui.py`|WebUI 应用；用于交互式测试与演示。|
|`start_webui.sh`|启动 WebUI 的便捷脚本。|
|`eval_local.py`|本地评测脚本：请求 `/predict` 并按评测口径计算 RougeL-F1（jieba 分词）。|
|`judge.sh`|近似评测机调用流程的本地评测脚本。|
|`data.jsonl`|评测/训练用 Q&A 数据集。|
|`metax_url.json`|运行/调试时用到的服务地址或平台相关配置。|
|`auto_tune.py`|自动调参主程序：在约束下搜索推理参数组合，平衡准确率与吞吐。|
|`auto_tune.sh`|自动调参的 Shell 包装脚本（启动/传参/日志路径等）。|
|`run_autotune_forever.sh`|持续循环运行自动调参（便于长时间搜索最优配置）。|
|`autotune.service.example`|systemd 服务示例：将自动调参以守护进程方式运行。|
|`merge_adapter.py`|将 LoRA/PEFT adapter 融合进基座模型并导出 merged 权重。|
|`upload_model.py`|将本地（HF 格式）模型目录上传到 ModelScope。|
|`quantize_awq.py`|AWQ 量化脚本（基于AutoAWQ）。|
|`sample_calib_from_data.py`|从 `data.jsonl` 抽样生成量化所需的校准数据（calibration set）。|
|`calib_512.jsonl` / `calib_512.txt`|量化校准数据（512 规模/上下文版本，供 AWQ 使用）。|
|`calib_8192.jsonl` / `calib_8192.txt`|量化校准数据（8192 规模/上下文版本，供 AWQ使用）。|
|`requirements.txt`|**在线服务最小依赖集合**（包含 `serve.py`/`download_model.py` 所需）。|
|`requirements-eval.txt`|本地评测所需依赖集合。（包含 `eval_local.py` 所需）。|
|`requirements-merge.txt`|融合 adapter（PEFT/transformers 等）所需依赖集合。（包含 `merge_adapter.py` 所需）。|
|`requirements-quantize-awq.txt`|AWQ 量化所需依赖集合。（包含 `quantize_awq.py` 所需）。|
|`requirements-webui.txt`|WebUI 依赖集合（包含 `webui.py` 所需）。|
|`tune_secrets.example.sh`|调参/上传等需要的密钥环境变量示例（需要填充密钥才能使用）。|
|`basic.docx` / `plus.docx`|项目使用的资料/原始文档（用于测评调用）。|

---

以下内容是旧文档，暂未完全重排（可作为补充参考）：

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

1. 安装系统依赖（需要输入 sudo 密码）：

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip
```

1. 创建并激活虚拟环境：

```bash
cd /path/to/metax-llm-public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

1. 安装 merge 所需 Python 依赖（推荐用最小集合）：

```bash
pip install -r requirements-merge.txt
```

1. 安装 PyTorch

- CPU-only（WSL/无 GPU 最稳）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

1. 运行融合：

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

1. 启动服务并等待健康检查通过

2. 跑 `eval_local.py` N 次并计算均值

3. 关闭服务并等待端口释放

4. 把本轮结果写入 `tune_results.jsonl`

只有完成（或明确失败并记录）后才会进入下一轮。

- 优化目标：在 `Accuracy >= 阈值` 前提下，最大化 `Throughput RAW: (prompt+answer)_tokens/s`。

- 结果输出：

  - `tune_results.jsonl`：每个试验一行（含均值、日志路径、失败原因等）

  - `best_params.json`：实时保存当前最优参数（后续试验会基于它继续搜索）

  - `tune_status.json`：实时状态（当前测试参数/进度），便于外部轮询查看

运行示例：

```bash
cd /data/metax-llm-public
source ./env_force.sh
./auto_tune.sh
```

### 容器环境保活（无 systemd / 无 systemctl）

很多云平台是在容器内运行（PID 1 不是 systemd），此时 `systemctl` 会报：

> System has not been booted with systemd as init system...

请使用本仓库提供的 watchdog：服务异常退出就自动重启；同时 `auto_tune.py` 会基于 `tune_results.jsonl` 自动断点续跑。

1. 更新代码并准备私密配置：

```bash
cd /data/metax-llm-public
git pull

cp tune_secrets.example.sh tune_secrets.sh
vim tune_secrets.sh
chmod 600 tune_secrets.sh
```

1. 可选：无模型自检（不启动模型，只校验解析/落盘/通知逻辑）：

```bash
./auto_tune.py --selftest --repo .
```

1. 后台启动 watchdog（断线不掉）：

```bash
nohup ./run_autotune_forever.sh > autotune.watchdog.log 2>&1 &
```

如需每 10 分钟发一次“我还活着”的飞书/邮件心跳（含已运行时长），在运行前设置：

```bash
export TUNE_HEARTBEAT_INTERVAL_S=600
# 可选：关闭“按 trial 次数心跳”（默认每 10 个 trial 一次）
export HEARTBEAT_TRIALS=0
```

1. 观察与停止：

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

1. Webhook（推荐，通用）：

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

1. SMTP 邮件：

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

  - 可调：`SPEC_NUM_SPECULATIVE_TOKENS=6`（建议 4\~8）

- 构建期下载 draft（Dockerfile 已透传参数，默认 draft 下载失败不阻断构建）：

  - `SPEC_DRAFT_MODEL_ID=<ModelScope 模型 ID>`

  - `SPEC_DRAFT_MODEL_REVISION=master`

注意：vLLM 当前实现中 speculative decoding 与 chunked prefill 不兼容；当启用 speculative 时，服务端会强制关闭 `enable_chunked_prefill`。

## AWQ 量化（AutoAWQ，覆盖上传同名模型）

说明：部分环境/架构不支持 Marlin kernel，因此此前 compressed-tensors 路线可能无法运行。这里提供 AutoAWQ 量化脚本，输出目录固定为 `model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ`，可直接用上传脚本覆盖同名仓库。

1. 安装量化依赖（建议单独虚拟环境；不修改线上 serving 的 requirements.txt）：

```bash
python -m venv .venv-awq
source .venv-awq/bin/activate

pip install -U pip setuptools wheel

# 先安装 transformers 等轻依赖（避免 autoawq 自动升级 torch 破坏环境）
pip install -r requirements-quantize-awq.txt

# 再安装 autoawq（不自动拉取/升级 torch）
pip install --no-deps autoawq==0.2.9
```

1. 量化并导出到固定目录：

```bash
python quantize_awq_llmcompressor.py \
  --model_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM \
  --calib_jsonl calib_512.jsonl \
  --output_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ
```

1. 上传并覆盖 ModelScope 上同名仓库：

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
|-|-|
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