# 微调Qwen大模型的推理部署与工具

**2025年秋季中国科学院大学(国科大)《GPU架构与编程》课程项目二**

📖 详细文档请查看：<https://docs-gpu.yukino.uk>

## 📌 导航

- 🚀 快速开始

- 🧭 使用说明

- 🗂️ 文件/目录说明

---

## 🏆 摩尔线程一等奖——沐曦赛道

---

本项目提供了 vLLM 加速的推理代码和 Web 界面以及一些测试模型的工具脚本。

Gitee仓库（用于实际提交）：<https://gitee.com/yukinostuki/metax-demo>

Github仓库（用于具体开发）：<https://github.com/YukinoStuki2/metax-llm-public>

两个仓库仅有文件数量区别，其版本和代码内容都是同步的。

## 📚 项目介绍

这是一个对微调开源大模型进行推理，用于问答评测的项目：

- 微调数据集来源于《Programming Massively Parallel Processors.2017》。

- 评测目标：在准确率（RougeL-F1，jieba 分词）达到阈值（常见参考 ≥ 0.35）的前提下，尽量提升吞吐（tokens/s）。

评测系统关键约束：

- Build 阶段允许联网（用于下载依赖/权重）；Run 阶段断网（请求路径内不做任何联网操作）。

- 必须提供：

  - `GET /`：健康检查（返回“ok”或“batch”切换模式）

  - `POST /predict`：推理接口（见下文 API 契约）

- 端口：保持 `8000`。

- `GET /` 返回 `{"status":"batch"}`，评测系统会将所有问题一次性推到 `POST /predict`。

  * batch 请求格式：`{"prompt":["...","...",...]}`

  * batch 响应格式：`{"response":["...","...",...]}`（答案数量需与问题数量一致）

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

### 🌐 使用 WebUI（推荐）

导入环境变量：

```bash
source ./env_force.sh
```

启动推理后端：

```bash
bash ./run_model.sh
```

启动 Web 界面：

```bash
./start_webui.sh
```

浏览器访问：<http://localhost:7860>

（可选）如果在沐曦服务器中启动，可以通过 SSH 隧道连接 WebUI：

```shell
ssh -CNg -L 7860:127.0.0.1:7860 root+<username>@<IP> -p <PORT>
```

### 🐳 使用 Docker 启动（评测系统无 WebUI，不保证一定能正常运行，我没评测机的具体程序）

**注意：沐曦容器上无法直接用 Docker 启动。**

本地自测：

```bash
docker build -t metax-demo:latest .
docker run --rm -p 8000:8000 metax-demo:latest
```

时间限制（参考）：

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

## 🧭 使用说明

使用本仓库的一些脚本可以便捷地完成各种操作。

### ✅ WebUI

适合本地/云主机交互式测试（评测机 Docker 运行不包含 WebUI）。

浏览器打开：[http://localhost:7860 ](http://localhost:7860)

更多功能（生成参数、SYSTEM_PROMPT、RAG、Batch 测试）见：[README_WEBUI.md](README_WEBUI.md)

### ✅ 切换模型

默认会下载/加载 `MODEL_ID` 对应的模型，运行目录默认是 `MODEL_DIR=./model/$MODEL_ID`。

切换到另一个 ModelScope 模型：

```bash
source ./env_force.sh
MODEL_ID=YukinoStuki/Qwen3-4B-Plus-LLM MODEL_REVISION=master bash ./run_model.sh
```

切到本地目录（已提前准备好权重，避免启动时下载）：

```bash
source ./env_force.sh
MODEL_DIR=./model/YukinoStuki/Qwen3-4B-Plus-LLM SKIP_MODEL_DOWNLOAD=1 bash ./run_model.sh
```

可供选择的模型有：

稳定，可评测可使用：

[YukinoStuki/Qwen2.5-0.5B-Plus-LLM （最快速）](https://modelscope.cn/models/YukinoStuki/Qwen2.5-0.5B-Plus-LLM)

[YukinoStuki/Qwen3-4B-Plus-LLM](https://modelscope.cn/models/YukinoStuki/Qwen3-4B-Plus-LLM)[（最智能）](https://modelscope.cn/models/YukinoStuki/Qwen2.5-0.5B-Plus-LLM)

不稳定，可使用但评测可能过不了：

[YukinoStuki/Qwen2.5-0.5B-Plus-EN （英语回复）](https://modelscope.cn/models/YukinoStuki/Qwen2.5-0.5B-Plus-EN)

[YukinoStuki/Qwen3-0.6B-Plus-LLM （较快速）](https://modelscope.cn/models/YukinoStuki/Qwen3-0.6B-Plus-LLM)

[YukinoStuki/Qwen3-1.7B-Plus-LLM （快速）](https://modelscope.cn/models/YukinoStuki/Qwen3-1.7B-Plus-LLM)

[YukinoStuki/Qwen2.5-1.7B-Plus-LLM（快速）](https://modelscope.cn/models/YukinoStuki/Qwen2.5-1.7B-Plus-LLM)

[YukinoStuki/Qwen3-4B-Plus-LLM-AWQ （量化-低精度）](https://modelscope.cn/models/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ)

[YukinoStuki/Qwen2.5-0.5B-Plus-AWQ](https://modelscope.cn/models/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ)[（量化-超低精度）](https://modelscope.cn/models/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ)

[YukinoStuki/Qwen2.5-0.5B-Plus-CCC （不同数据集）](https://modelscope.cn/models/YukinoStuki/Qwen2.5-0.5B-Plus-CCC)

直接把MODEL_ID和MODEL_DIR更换为以上模型的完整名字即可，如MODEL_ID=YukinoStuki/Qwen3-4B-Plus-LLM

### ✅ 切换参数

最常用的推理参数（启动前覆盖即可）：

```bash
# 刷新全部默认变量
source ./env_force.sh
# 控制输出长度 / 吞吐
MAX_NEW_TOKENS=128 MAX_MODEL_LEN=4096 \
# 解码稳定性
TEMPERATURE=0.0 TOP_P=1.0 TOP_K=1 \
# 显存占用
GPU_MEMORY_UTILIZATION=0.90 \
bash ./run_model.sh
```

batch 开关：

```bash
# 开启 batch（默认通常已开）
BATCH_MODE=1 BATCH_CONCURRENCY=512 bash ./run_model.sh

# 关闭 batch（单条请求模式）
BATCH_MODE=0 bash ./run_model.sh
```

所有可更改的参数请参照详细文档<https://docs-gpu.yukino.uk>

### ✅ 直接运行模型

启动后端（FastAPI + uvicorn，端口固定 8000）：

```bash
source ./env_force.sh
uvicorn serve:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/
```

单条推理：

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"请简要回答：什么是CUDA？"}'
```

### ✅ 对模型进行评测

项目提供多参数评测脚本[eval_local.py](eval_local.py)，能在本地一定程度上模拟评测机得分

评测脚本会输出准确率和token/s速度，不一定准确！仅参考意义（比如我知道我这次跑的比上次好即可）

评测脚本会调用后端 `/predict`，所以**先启动后端**再跑评测。

方式一：使用封装脚本（出问题了请用方法二）：

```bash
source ./env_force.sh
bash ./run_model.sh

# 新开终端
MODEL_DIR="$MODEL_DIR" WHICH=bonus bash ./judge.sh
```

方式二：直接跑 `eval_local.py`（basic是基础题，bonus是加分题，修改which参数）：

```bash
python3 eval_local.py \
  --which bonus \
  --batch \
  --overwrite_jsonl \
  --model_dir_for_tokenizer "$MODEL_DIR"
```

eval_local.py有很多很多参数可以调控，请参照详细文档<https://docs-gpu.yukino.uk>

### ✅ 启用推理解码

本项目支持 speculative decoding（默认关闭）。ngram 方法不需要额外 draft 模型，零额外权重成本：

```bash
source ./env_force.sh
ENABLE_SPECULATIVE_DECODING=1 SPEC_METHOD=ngram SPEC_NUM_SPECULATIVE_TOKENS=6 bash ./run_model.sh
```

如需 draft 模型（沐曦vllm0.10.0不支持draft）：

```bash
ENABLE_SPECULATIVE_DECODING=1 \
SPEC_METHOD=draft \
SPEC_DRAFT_MODEL_ID=<draft_model_id> \
SPEC_NUM_SPECULATIVE_TOKENS=6 \
bash ./run_model.sh
```

### ✅ 量化

仓库内提供 AWQ 量化脚本 [quantize_awq.py](quantize_awq.py)（AutoAWQ 4bit，参数内置偏保准确率；**单独虚拟环境安装量化依赖**，AutoAWQ和vllm冲突）。

1）安装量化依赖：

```bash
python3 -m venv .venv-awq
source .venv-awq/bin/activate
python -m pip install -U pip setuptools wheel

# 先安装 transformers 等轻依赖（避免 autoawq 自动升级 torch 破坏环境）
python -m pip install -r requirements-quantize-awq.txt

# 再安装 autoawq
python -m pip install autoawq==0.2.9
```

2）生成校准集（`quantize_awq.py` 期望 jsonl 每行 `{"text":"..."}`；示例：8192 条，最大长度 2048 字符）：

```bash
N=8192 MAX_LEN=2048 OUT_JSONL=calib_8192.jsonl OUT_TXT=calib_8192.txt python3 sample_calib_from_data.py
```

3）执行量化导出：

```bash
# 可用 AWQ_CALIB_JSONL 覆盖校准集路径；脚本默认值目前是 calib_8192.jsonl
AWQ_CALIB_JSONL=calib_8192.jsonl \
python3 quantize_awq.py \
  --model_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM \
  --output_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ
```

4）加载 AWQ 模型启动后端（vLLM 侧透传）：

```bash
source ./env_force.sh

# 使用量化后的本地目录；如已生成无需下载
MODEL_DIR=./model/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ \
SKIP_MODEL_DOWNLOAD=1 \
VLLM_QUANTIZATION=awq VLLM_LOAD_FORMAT=awq \
bash ./run_model.sh
```

### ✅ 上传模型到 ModelScope

项目提供模型上传脚本 [upload_model.py](upload_model.py)，用于把**本地模型目录**上传到 ModelScope 指定仓库（用于把量化/融合后的结果覆盖上传到同名 repo）。

脚本参数规则（与代码一致）：

- `--repo-id`：目标 ModelScope 仓库 ID；也可用环境变量 `REPO_ID` 覆盖。

- `--model-dir`：本地模型目录；也可用环境变量 `MODEL_DIR` 覆盖。

- token：优先用 `--token`；否则读取环境变量 `MODELSCOPE_API_TOKEN` 或 `MODELSCOPE_TOKEN`。

示例 1：上传量化后的 0.5B AWQ（覆盖同名仓库）：

```bash
export MODELSCOPE_API_TOKEN='<your_token>'

python3 upload_model.py \
  --repo-id YukinoStuki/Qwen2.5-0.5B-Plus-AWQ \
  --model-dir model/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ \
  --commit-message 'overwrite awq (autoawq)'
```

示例 2：用环境变量简化（脚本会读 `REPO_ID`/`MODEL_DIR`）：

```bash
export MODELSCOPE_API_TOKEN='<your_token>'
export REPO_ID='YukinoStuki/Qwen2.5-0.5B-Plus-AWQ'
export MODEL_DIR='model/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ'

python3 upload_model.py --commit-message 'upload model folder'
```

### ✅ 自动调参（auto_tune）

项目提供自动调参脚本 [auto_tune.py](auto_tune.py) 与启动封装 [auto_tune.sh](auto_tune.sh)。它会循环尝试不同参数组合：

- 启动后端（8000）→ 健康检查 → 多次运行 `eval_local.py` → 取平均分/速度 → 记录结果 → 关闭服务

- 失败/超时会自动跳过进入下一组组合（脚本不中断）

- 支持断点续跑（默认会跳过已跑过的组合）

1）准备（推荐先清“干净参数”）：

```bash
source ./env_force.sh
```

2）配置通知/密钥（可选，可以通过**飞书**或者**邮箱**发送通知）：

```bash
cp tune_secrets.example.sh tune_secrets.sh
chmod 600 tune_secrets.sh
# 然后编辑 tune_secrets.sh，填入飞书 webhook 或 SMTP 等配置
```

说明：`auto_tune.sh` 启动时会自动 `source ./tune_secrets.sh`（若存在）。

3）启动自动调参：

```bash
./auto_tune.sh
```

常用覆盖参数（直接写在命令前即可）：

```bash
# ACC: 准确率阈值；EVAL_RUNS: 每组参数重复评测次数
ACC=0.8810 EVAL_RUNS=5 ./auto_tune.sh

# 超时控制（启动/评测）
STARTUP_TIMEOUT=240 EVAL_TIMEOUT=420 ./auto_tune.sh

# 心跳通知（每 N 个 trial 或每隔 N 秒发一次；0=关闭）
HEARTBEAT_TRIALS=10 TUNE_HEARTBEAT_INTERVAL_S=600 ./auto_tune.sh
```

输出产物（默认写在仓库根目录）：

- `tune_results.jsonl`：每个参数组合一行结果（断点续跑会用它去重）

- `best_params.json`：当前最优参数快照

- `tune_status.json`：实时状态（便于外部监控）

- `tune_server_logs/`：每次试验的服务端日志

高级：扩展搜索空间（可选）

- 新建一个 JSON 文件，例如：`tune_search_space.json`，内容形如：`{"GPU_MEMORY_UTILIZATION":["0.95","0.97"],"MAX_MODEL_LEN":["4096","8192"]}`

- 启动时指定：

```bash
TUNE_SEARCH_SPACE_FILE=./tune_search_space.json ./auto_tune.sh
```

端口占用处理（可选）：默认只等待重试，不会杀进程；若你确认占用者是残留 `uvicorn serve:app`，可谨慎开启：

```bash
TUNE_PORT_BUSY_KILL=1 ./auto_tune.sh
```

守护运行（可选）：

- 无 root：用 [run_autotune_forever.sh](run_autotune_forever.sh) 让 `auto_tune` 异常退出自动重启

- 有 systemd：参考 [autotune.service.example](autotune.service.example) 作为服务部署

注意：通知（飞书/邮件）需要联网。

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
|`README.md`|GitHub 主 README。|
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
|`quantize_awq.py`|AWQ 量化脚本（AutoAWQ 4bit；偏保准确率的内置参数）。|
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

以下内容是旧文档，暂未完全重排（暂时忽略以下内容）：

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

## 