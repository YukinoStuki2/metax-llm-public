---
sidebar_position: 4
---

# 文件与目录说明（全仓库）

本页按照仓库当前结构，对所有文件/目录的用途做一份“全量索引”。阅读建议：

- 如果你的目标是“评测交付 / Gitee 精简仓库”：重点关注“评测交付最小集”一节。
- 如果你的目标是“本地开发/调参/量化/写文档”：按功能模块浏览后续分组。

说明：仓库中存在一些“本地生成物/缓存目录”（例如 `model/`、`.venv/`、`docs-site/build/`）。这类内容通常不需要提交，也不应被评测链路依赖。本页会明确标注。

---

## 评测交付最小集（用于镜像构建与推理服务）

这部分文件原则上足以完成评测机的 build/run：

|路径|用途|
|---|---|
|`Dockerfile`|镜像构建入口：安装依赖、下载模型、设置默认环境变量，并启动推理服务（端口 8000）。|
|`requirements.txt`|在线服务最小依赖集合（供 `serve.py` 与 `download_model.py` 使用）。|
|`download_model.py`|build 阶段从 ModelScope 下载模型权重到 `./model/`；可选下载 speculative decoding 的 draft 模型。|
|`serve.py`|推理服务核心：FastAPI 应用，必须提供 `GET /` 与 `POST /predict`；run 阶段断网，禁止在请求路径联网。|
|`README_GITEE.md`|面向评测提交的精简说明（同步到 Gitee 时会覆盖为 README）。|
|`data.jsonl`|可选数据：用于预热/抽样/调试；不应成为服务启动强依赖。|

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
|`项目情况.txt`|项目说明/备忘（非评测必需）。|

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
|`basic.docx` / `plus.docx`|题目/素材的原始文档（历史资料）。|

---

## 文档站（docs-site/）

`docs-site/` 是 Docusaurus 文档站源码与构建产物，在线文档来源于这里。

|路径|用途|
|---|---|
|`docs-site/package.json`|文档站前端依赖与脚本命令。|
|`docs-site/docusaurus.config.js`|站点配置（标题、导航、部署配置等）。|
|`docs-site/sidebars.js`|侧边栏结构配置。|
|`docs-site/README.md`|文档站开发说明（Docusaurus 模板自带）。|
|`docs-site/docs/`|文档正文（Markdown）。|
|`docs-site/src/`|站点 React 组件与页面源码。|
|`docs-site/static/`|静态资源（图片等），构建时原样拷贝。|
|`docs-site/build/`|站点构建输出（生成物；通常不建议手工编辑）。|

`docs-site/docs/` 内按主题拆分了脚本文档与说明页（例如 service、tuning、quant、webui 等）；如要新增脚本说明，建议按现有目录结构新增页面。

---

## GitHub 自动化与仓库元信息

|路径|用途|
|---|---|
|`.github/workflows/sync_to_gitee.yml`|GitHub Actions：每次 push 到 master 后裁剪文件树并强制同步到 Gitee（评测提交仓库）。|
|`.github/copilot-instructions.md`|Copilot 工作约束（评测契约、参数对齐规则、依赖添加约束等）。|
|`.git/`|Git 元数据目录。|

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
|`docs-site/build/`|Docusaurus 构建输出（生成物）。|

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
