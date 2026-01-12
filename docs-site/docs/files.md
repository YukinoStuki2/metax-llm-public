---
sidebar_position: 4
---

# 文件与目录说明（全仓库）

本页按功能分组说明仓库内各文件/脚本的用途，方便“看到文件名就知道该看哪份文档/该跑哪个脚本”。

说明：你本地可能会出现一些“生成物/缓存目录”（例如 `model/`、`.venv/`、`__pycache__/`）。这类内容通常不需要提交，也不应成为推理服务的运行时强依赖。

---

## 根目录（入口脚本与常用文档）

|路径|用途|
|---|---|
|`.dockerignore`|控制 Docker build 上下文的包含/排除，减少体积与构建时间。|
|`.gitignore`|Git 忽略规则（模型、日志、虚拟环境、调参产物等）。|
|`README.md`|GitHub 主 README：快速开始、使用说明、文件/目录说明（较长）。|
|`README_WEBUI.md`|WebUI 的功能说明与常见问题。|
|`DEPLOY.md`|部署相关说明（环境、启动方式、注意事项等）。|
|`QUICKREF.md`|常用命令速查（启动、评测、调参、量化等）。|
|`README_GITEE.md`|精简 README（用于同步/镜像提交场景）。|

---

## 启动与环境变量管理（.sh）

|路径|用途|
|---|---|
|`run_model.sh`|在 Docker 外复现“安装依赖→下载模型→启动服务”的最小启动脚本；仅在未设置变量时填默认值。|
|`env_force.sh`|强制导入一套“干净参数”（覆盖当前 shell 的同名变量）；必须用 `source` 执行。|

---

## 推理服务与交互（后端 + WebUI）

|路径|用途|
|---|---|
|`serve.py`|后端推理服务：FastAPI + uvicorn；支持单条与 batch 推理。|
|`webui.py`|Gradio WebUI：通过 HTTP 调用后端 `/predict`；包含可选 RAG、本地 batch 测试、system prompt 管理、/info 展示等。|
|`start_webui.sh`|WebUI 启动器：创建/激活 venv，安装 WebUI+评测依赖，探活后端并启动 `webui.py`。|
|`metax_url.json`|WebUI 的固定 URL 库（用于可选 RAG：从种子 URL 中选取候选并抓取内容）。|

---

## 本地评测与评测模拟

|路径|用途|
|---|---|
|`eval_local.py`|本地评测脚本：请求后端 `/predict`，按评测口径计算 RougeL-F1（jieba 分词）并统计吞吐。|
|`judge.sh`|封装式本地评测流程：更接近评测机调用方式（会按固定口径跑评测并输出结果）。|
|`eval_details.jsonl`|评测结果/明细记录（通常为本地产物；是否提交取决于你的工作流）。|

---

## 自动调参（auto_tune）

|路径|用途|
|---|---|
|`auto_tune.py`|自动调参主程序：循环启动服务→健康检查→多次跑评测→记录结果→停止服务；支持断点续跑与通知。|
|`auto_tune.sh`|自动调参启动封装：读取常用环境变量并映射为 `auto_tune.py` 参数；可自动 source `tune_secrets.sh`（若存在）。|
|`run_autotune_forever.sh`|守护脚本：`auto_tune` 异常退出时自动重启（无 systemd 场景）。|
|`autotune.service.example`|systemd 服务示例：把自动调参作为服务运行与重启。|
|`tune_secrets.example.sh`|通知/密钥配置示例（飞书 webhook、SMTP 等）；复制为 `tune_secrets.sh` 并填值使用。|
|`tune_secrets.sh`|本地密钥文件（通常不应提交；权限建议 `chmod 600`）。|
|`tune_server_logs/`|调参过程中每次 trial 的服务端日志目录（本地生成物）。|

---

## 模型工程（下载/融合/上传）

|路径|用途|
|---|---|
|`download_model.py`|从 ModelScope 下载模型权重到指定目录（build 阶段使用为主）。|
|`merge_adapter.py`|将 LoRA/PEFT adapter 融合进基座模型并导出 merged 权重。|
|`upload_model.py`|将本地模型目录上传到 ModelScope 指定仓库（用于覆盖上传量化/融合产物）。|

---

## 量化（AWQ）与校准集

|路径|用途|
|---|---|
|`quantize_awq.py`|AWQ 量化脚本（AutoAWQ 4bit；通常需单独虚拟环境安装量化依赖）。|
|`sample_calib_from_data.py`|从 `data.jsonl` 抽样生成量化所需校准集（jsonl 每行 `{"text":"..."}`）。|
|`calib_512.jsonl` / `calib_512.txt`|示例校准集（较小规模），供量化调试。|
|`calib_8192.jsonl` / `calib_8192.txt`|示例校准集（较大规模），供量化使用。|

---

## 依赖清单（requirements-*.txt）

这些文件用于把不同功能的依赖隔离开（避免互相冲突）：

|路径|用途|
|---|---|
|`requirements.txt`|服务端最小依赖（`serve.py`/`download_model.py`）。|
|`requirements-eval.txt`|本地评测依赖（`eval_local.py`）。|
|`requirements-webui.txt`|WebUI 依赖（`webui.py`）。|
|`requirements-merge.txt`|融合 adapter 依赖（`merge_adapter.py`）。|
|`requirements-quantize-awq.txt`|AWQ 量化相关依赖（`quantize_awq.py`；通常不含 autoawq 本体以避免升级 torch）。|

---

## 数据与素材

|路径|用途|
|---|---|
|`data.jsonl`|问答数据集（用于评测/预热/抽样等）。|
|`basic.txt` / `plus.txt`|题目/素材的文本版本（用于调试或对照）。|

---

## 本地生成物/缓存目录（通常不需要提交）

这些目录可能出现在你的工作区中，但一般不建议作为“交付物”依赖：

|路径|用途|
|---|---|
|`model/`|本地模型权重目录（`download_model.py` 下载、量化/融合产物等）；体积大，通常不提交。|
|`.venv/`|WebUI/服务本地虚拟环境（本地生成物）。|
|`.venv-awq/`|量化专用虚拟环境（避免依赖冲突）。|
|`__pycache__/`|Python 字节码缓存。|
|`tune_server_logs/`|自动调参产生的服务端日志（本地生成物）。|

---

## 进一步阅读（文档站入口）

为了避免单页过长，仓库的主要入口脚本均有单独文档页：

- 推理服务：`serve.py`
- 启动脚本：`run_model.sh` / `env_force.sh`
- WebUI：`webui.py` / `start_webui.sh`
- 本地评测：`eval_local.py` / `judge.sh`
- 自动调参：`auto_tune.py` / `auto_tune.sh`
- 模型工程：`download_model.py` / `merge_adapter.py` / `upload_model.py`
- 量化：`quantize_awq.py` / `sample_calib_from_data.py`
