---
title: 推理服务（代码详解）
sidebar_position: 12
---

本页对 `serve.py` 做“按模块/按函数”的实现级讲解，目标是：你能在不读完整 1700+ 行代码的情况下，准确理解它为什么这么写、哪里能改、改了会影响什么。

- 用法与定位：看 [推理服务（用法与定位）](./serve)
- 参数与环境变量：看 [推理服务（参数与环境变量）](./serve_env)
- 常见问题：看 [推理服务（FAQ）](./serve_faq)

## 总体结构

`serve.py` 可以粗分为 6 层：

1) **配置区（全局常量 + env 读取）**：模型目录、默认生成参数、token routing 规则、预热参数、stop 规则。
2) **题型判定与长度分流**：`is_code_question / is_hard_code_question / is_long_answer_question / pick_max_new_tokens`。
3) **prompt 组装与输出后处理**：`get_system_prompt / set_system_prompt / format_as_chat / strip_think / _postprocess_answer`。
4) **后端选择与兼容**：可选导入 vLLM、vLLM v1、torch/transformers；`should_use_vllm`。
5) **应用生命周期**：`lifespan(app)` 在启动阶段加载模型、初始化引擎并预热。
6) **HTTP 路由**：`GET /`、`GET /info`、`GET/POST /system_prompt`、`POST /predict`。

## 1) 配置区：为什么大量用环境变量

服务面向“评测容器”，而评测容器的现实约束是：

- build 阶段可以联网下载模型；run 阶段断网。
- 不希望在请求路径上做网络探测。
- 需要通过 Dockerfile / 启动脚本快速调参。

因此 `serve.py` 的大部分行为都由环境变量控制，并尽量做到：

- **默认值安全**：没配也能跑。
- **失败可回退**：vLLM 不可用时回退 transformers（除非 `FORCE_VLLM=1`）。
- **跨平台兼容**：对 vLLM 参数采用“签名存在才注入”的方式，避免不同构建/平台插件直接启动失败。

## 2) 题型判定与 token routing

### `_normalize_text`
- 做最小清洗：`strip().lower()`。

### `is_code_question(user_prompt)`
- 启发式判断题目是否“明确要求代码/伪代码/实现”。
- 设计目标是“保守”：避免把包含 CUDA 术语（如 `threadIdx`）的概念题误判为代码题。
- 支持通过环境变量 `CODE_QUESTION_KEYWORDS` 追加关键词。

### `is_hard_code_question(user_prompt)`
- 更强信号：只有命中 `kernel<<<`、`global void`、`#include` 等（以及 `HARD_CODE_QUESTION_KEYWORDS` 追加）才算。
- 通过 `HARD_CODE_MIN_HITS` 控制最小命中数（并做 1~5 的裁剪）。

### `is_long_answer_question(user_prompt)`
- 识别“参考答案通常较长”的题（经验上集中在 bonus/plus 的算子类问答）。
- 默认启用内置关键词（`LONG_ANSWER_ENABLE_DEFAULT=1`），同时支持 `LONG_ANSWER_KEYWORDS` 追加。
- `LONG_ANSWER_MIN_HITS` 控制最小命中数（1~5）。

### `pick_max_new_tokens(user_prompt)`
- 分流的最终入口：
  - 若 `DISABLE_TOKEN_ROUTING`：恒等于 `MAX_NEW_TOKENS`。
  - hard code：`MAX_NEW_TOKENS_CODE_HARD`
  - long answer：`MAX_NEW_TOKENS_CODE`
  - code：`MAX_NEW_TOKENS_CODE_SOFT`
  - 默认：`MAX_NEW_TOKENS`

这套分流逻辑的核心取舍：

- **短答题**尽量短（吞吐高、也更贴近教材短句风格）。
- 对少量“必须长”的题拉长，避免截断导致 Rouge 掉分。

## 3) prompt 组装与输出约束

### 系统提示词：`SYSTEM_PROMPT` / `get_system_prompt()` / `set_system_prompt()`
- `SYSTEM_PROMPT` 是评测导向的提示词：强调“只输出答案正文”“短句覆盖关键点”“不要分析/步骤”。
- `_SYSTEM_PROMPT_CURRENT` 是运行时可变版本，WebUI 可通过 API 动态更新。
- `set_system_prompt` 会 `strip()` 并对长度做 20000 的防御性上限。

### `build_prompt(user_prompt)`
- 旧的纯文本提示格式（system + “问题/答案”），用于 tokenizer 不可用时兜底。

### `format_as_chat(tokenizer, user_prompt)`
- 首选用 Qwen 系列的 chat template 构造 prompt，以提升一致性和准确率。
- 为了 batch 场景吞吐：
  - 如果 `FAST_CHAT_TEMPLATE` 启用且检测到模板包含 `<|im_start|>`，走字符串拼接快路径。
  - 否则回退到 `tokenizer.apply_chat_template(...)`。
- 若 tokenizer 不可用，则回退到 `build_prompt`。

### `<think>` 剥离：`strip_think(text)`
- 评测要求避免输出很长的“思考过程”。
- 但有些模型会把最终答案也写在 `<think>` 里，因此实现做了三层安全回退：
  1) 优先移除整个 `<think>...</think>` 块。
  2) 若移除后变空，尝试提取 think 内文本。
  3) 若仍为空，只去掉标签本身。

### 后处理：`_postprocess_answer(text, user_prompt)`
- 仅对“短答模式”（`pick_max_new_tokens(user_prompt) == MAX_NEW_TOKENS`）启用。
- 主要做两件事：
  - 裁剪“例如/比如/举例：”之后的扩展（默认开，`OUTPUT_TRIM_EXAMPLES=0` 可关）。
  - 最多保留 N 个句子片段（默认 6，`OUTPUT_MAX_SENTENCES` 可调）。

## 4) vLLM / Transformers 可选导入与选择

### 可选依赖探测
- vLLM：尝试导入 `SamplingParams / AsyncEngineArgs / AsyncLLMEngine`。
- vLLM v1：尝试导入 `vllm.v1.engine.async_llm.AsyncLLM`（可能不存在）。
- transformers：尝试导入 `torch / AutoTokenizer / AutoModelForCausalLM`。

### `should_use_vllm()`
- 即使 vLLM 可导入，也不一定选它：
  - 若没有 C 编译器（Triton 可能需要编译），并且没有 `FORCE_VLLM=1`，则直接用 transformers。
  - 若 `USE_VLLM=true`：强制尝试 vLLM（失败可回退）。
  - 若 `USE_VLLM=false`：禁用 vLLM。
  - 否则：仅当 CUDA 可用才倾向 vLLM。

### MetaX / 插件安全逻辑
- 非 MetaX 机器默认设置 `VLLM_PLUGINS=""`，防止误加载平台插件。
- MetaX 设备上可能默认启用 `VLLM_USE_V1=1`（但服务端引擎选择用 `SERVE_VLLM_ENGINE`，避免与平台语义冲突）。
- 对 `CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES / VLLM_DEVICE` 做“空字符串则删除”的防御性处理。

## 5) `lifespan(app)`：启动阶段发生了什么

FastAPI 的 `lifespan` 会在服务启动时执行（评测 health check 也会触发），因此实现强调：

- 一切都必须能在断网环境工作。
- 预热失败不影响服务（打印后继续）。

启动流程如下：

1) **解析模型目录**：
- 如果用户没设 `MODEL_DIR` 且默认目录不存在但 `./merged` 存在，则用 `./merged`。
- 否则用 `MODEL_DIR`（或默认拼出的目录）。
- 若目录不存在：直接 `RuntimeError`。

2) **初始化 tokenizer（尽力而为）**：
- 如果 transformers 可用：`AutoTokenizer.from_pretrained(..., use_fast=False, trust_remote_code=True)`。
- tokenizer 初始化失败不阻塞（可以用纯文本提示继续跑）。

3) **后端选择**：
- 优先 vLLM，失败回退 transformers（除非 `FORCE_VLLM=1`）。

4) **vLLM 初始化（复杂但关键）**：
- 先构造 `engine_kwargs`（只放通用/稳定参数）。
- 再根据 `AsyncEngineArgs.__init__` 的签名，按“支持就注入”的方式逐项加入可选参数：
  - 量化/加载格式、KV cache dtype、prefix caching、tokenizer pool、并发容量参数等。
- MetaX 设备上默认采取更保守的 `DEFAULT_MAX_MODEL_LEN`、默认禁用 chunked prefill。
- speculative decoding：若启用且 vLLM 参数支持，会注入 `speculative_config`，并强制关闭 chunked prefill。

5) **vLLM 多路径选择**：
- batch 模式下默认改用离线 `LLM.generate(list_prompts)`（`engine_kind="llm"`）。
- 否则用异步 `AsyncLLMEngine`（`engine_kind="async_v0"`）或 v1 `AsyncLLM`（`engine_kind="async_v1"`）。
- 若离线 LLM 初始化失败，会自动回退到 `AsyncLLMEngine`。

6) **vLLM 初始化失败的自动回退策略**（按优先级）：
 - compilation_config 类型不兼容：自动尝试 `dict <-> JSON` 字符串转换一次。
- load_format 不兼容：自动移除后重试一次。
- MetaX 非 eager 失败：自动回退 `enforce_eager=True` 重试一次。
- KV cache 不足：解析异常中的 `estimated maximum model length is ...`，或用 `SAFE_MAX_MODEL_LEN` 重试一次。

7) **vLLM 预热**：
- 使用 chat template 构造 warm prompt（让 prefix cache 真正热起来）。
- 离线 LLM 路线下，支持加载 `WARMUP_DATA_PATH` 抽样若干 prompt 并批量预热（放到线程池）。

8) **transformers 初始化与预热**：
- `AutoModelForCausalLM.from_pretrained(..., device_map="auto" if cuda else None)`。
- 简单生成一次 `max_new_tokens=8` 作为预热。

## 6) 路由层：GET / 与 POST /predict 的关键点

### `GET /` 健康检查
- 未 ready：返回 `{"status":"warming"}`。
- `BATCH_MODE=1`：返回 `{"status":"batch"}`。
- 否则：返回 `{"status":"ok"}`。

### `GET /info`
- 给 WebUI 展示的轻量信息。
- 只回传环境变量白名单（避免泄露敏感信息）。

### `GET /system_prompt` / `POST /system_prompt`
- 运行时动态读取/更新系统提示词；更新后立即影响后续 `/predict` 的 prompt 组装。

### `POST /predict`
- 请求体 `prompt` 支持 `str` 或 `list[str]`。
- `max_new_tokens` 若在请求体提供，则**覆盖分流**；否则走 `pick_max_new_tokens`。
- 生成参数（temperature/top_p/top_k/repetition_penalty/frequency_penalty）可请求级覆盖。

#### vLLM + 离线 LLM 路线（batch 吞吐优先）
- 关键优化：**按 max_tokens 分桶**，每个桶用一个 `SamplingParams` 调一次 `LLM.generate(prompts)`。
- 这样可以避免“给 vLLM 传 per-prompt SamplingParams 列表”导致 batching 失效、tokens/s 暴跌。
- 返回前会 `strip_think`，并做 `_postprocess_answer`（短答模式才做）。

#### vLLM + AsyncEngine 路线（默认）
- 单条：直接 `run_one` 等待最终输出。
- batch：用 `asyncio.Semaphore(BATCH_CONCURRENCY)` 控制并发，把多个 `engine.generate(...)` 并发提交，触发引擎内 batching。

#### transformers 路线
- 单条：正常生成。
- batch：为兼容协议会串行生成（不建议在该路线下追求 batch 吞吐）。
- 通过 `eos_token_id` 注入 stop token id 列表；若 `text` 以 `prompt_text` 开头，会剥掉 prompt 前缀。

## 7) 进程启动

`__main__` 里固定：

- `uvicorn.run(app, host="0.0.0.0", port=8000)`

这与评测端口契约一致。

