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
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"请简要回答：什么是CUDA？"}'
```

### ✅ 对模型进行评测

项目提供多参数评测脚本[eval_local.py](eval_local.py)，能在本地一定程度上模拟评测机得分

评测脚本会输出准确率和token/s速度，以及一些额外选项。

**注意！！！！实际上这个脚本的输出和评测机得分有很大差别，我自己测试，脚本输出大于0.8800的准确率，能在评测机上得到大于0.35的准确率，而token/s速度也同样不准，5000的token在评测机上大概是20000的得分**

**这是因为，评测脚本所用的题目是助教所发的题目示例，本身模型已经过拟合了所以肯定得分高，此外评测机的token速度是清理了额外时间的，比如http请求时间，而这个本地脚本没有处理这一块**

**脚本本身的目的不是获得和评测机一模一样的得分，而是能够判断当前新的优化有没有用，以及对模型进行快速的测试。**

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
|`basic.docx` / `plus.docx`|基础题和加分题的示例。|

---
## 许可证

本项目遵循项目主许可证 (Apache-2.0)。

## 需要帮助?

- 📖 查看[详细文档](https://docs-gpu.yukino.uk)获取更多配置选项

- 🐛 提交 Issue 到 GitHub

- 📧 联系我的邮箱yukinostuki@qq.com