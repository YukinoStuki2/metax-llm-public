---
title: 推理服务（代码详解）
sidebar_position: 12
---

本页对 `serve.py` 的实现进行讲解。

- 服务启动：见 [推理服务（用法）](./serve)

- 参数与环境变量：见 [推理服务（参数与环境变量）](./serve_env)

- 常见问题：见 [推理服务（FAQ）](./serve_faq)

## 总体结构

`serve.py` 可以粗分为 6 层：

1. **配置区（全局常量 + env 读取）**：模型目录、默认生成参数、token routing 规则、预热参数、stop 规则。

2. **题型判定与长度分流**：`is_code_question / is_hard_code_question / is_long_answer_question / pick_max_new_tokens`。

3. **prompt 组装与输出后处理**：`get_system_prompt / set_system_prompt / format_as_chat / strip_think / _postprocess_answer`。

4. **后端选择与兼容**：可选导入 vLLM、vLLM v1、torch/transformers；`should_use_vllm`。

5. **应用生命周期**：`lifespan(app)` 在启动阶段加载模型、初始化引擎并预热。

6. **HTTP 路由**：`GET /`、`GET /info`、`GET/POST /system_prompt`、`POST /predict`。

---

## 1）配置区（全局常量 + env 读取）

### 职责

- 读取环境变量并给出默认值（Docker/评测机优先）。

- 定义生成策略默认值：短答优先、尽量减少无效 token、对特定题型放宽长度。

- 定义预热与提前停止（stop）策略，减少首请求延迟与无效尾巴。

### 核心常量/全局变量

#### 基础依赖与启动行为

- `os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")`

  - 目的：避免 `transformers` 在启动时引入 `torchvision`，降低 warning 噪音与潜在版本冲突。

#### 模型路径

- `MODEL_DIR`

  - 读取 `MODEL_DIR`；未设置时回退到 `./model/$MODEL_ID`（默认 `YukinoStuki/Qwen3-4B-Plus-LLM`）。

  - 这一策略与 `Dockerfile` / `download_model.py` 的默认下载路径对齐。

#### System Prompt（运行时可变）

- `SYSTEM_PROMPT`：默认系统提示词（短答、不要过程、不要客套）。

- `_SYSTEM_PROMPT_CURRENT`：**运行时可变**版本（WebUI 可热更新）。

对应函数：

- `get_system_prompt()`：读取 `_SYSTEM_PROMPT_CURRENT`，异常/非字符串时回退 `SYSTEM_PROMPT`。

- `set_system_prompt(new_prompt)`：更新 `_SYSTEM_PROMPT_CURRENT`；并做防御性限制（`<= 20000` 字符），避免误把超长 prompt 写进内存导致吞吐/显存/CPU 开销飙升。

#### 输出长度路由（token routing）

- `MAX_NEW_TOKENS`：短答默认上限（默认 32）。

- `MAX_NEW_TOKENS_CODE / _SOFT / _HARD`：代码/长答案题的不同上限。

- `DISABLE_TOKEN_ROUTING`：禁用路由时全部走 `MAX_NEW_TOKENS`（更简单、更稳，但可能截断 plus/代码题）。

#### “长答案题”开关与阈值

- `LONG_ANSWER_ENABLE_DEFAULT`：默认启用（通过关键词启发式分流）。

- `LONG_ANSWER_MIN_HITS`：命中关键字的最小次数。

#### 预热（Warmup）

- `WARMUP_PROMPT`：默认预热 prompt。

- `WARMUP_DATA_PATH / WARMUP_NUM_SAMPLES / WARMUP_REPEAT`：可选从本地数据集抽样预热。

  - 代码里对 `WARMUP_NUM_SAMPLES` 做了“防御性上限裁剪”，避免因误配置导致 health 阶段超时。

对应函数：

- `_load_warmup_user_prompts(path, limit)`

  - 支持 `jsonl` 与 `json` 两类输入。

  - 会尝试从字段 `messages`（role=user）或 `instruction/prompt/question/query/input` 提取用户 prompt。

  - 文件不存在/解析失败会返回空列表（保证启动健壮）。

#### Stop 规则（提前停止）

- `STOP_STRINGS`：逗号分隔；默认包含 Qwen 常见结束标记 `"<|im_end|>"`、`"<|endoftext|>"`。

- `STOP_ON_DOUBLE_NEWLINE`：可选把 `"\n\n"` 放入 stop，进一步缩短“尾巴”。

对应函数：

- `_build_stop_token_ids(tokenizer)`：从 tokenizer 里提取 `eos_token_id` 并尝试把常见结束 token 转成 id。

- `_build_sampling_params(tokenizer, max_tokens, ...)`：构建 vLLM `SamplingParams`：

  - 温度/TopP/TopK/惩罚/stop/stop_token_ids。

  - 注意：这里把 stop 策略同时在 **string stop** 和 **token stop** 两侧都尽力兼容。

#### Batch 模式与并发

- `is_batch_mode()`：运行时动态读取 `BATCH_MODE`（避免 import 固化导致 health 与实际不一致）。

- `BATCH_CONCURRENCY`：AsyncEngine 的 Python 层并发上限（通过 semaphore 控制）。

#### 采样与惩罚默认值

- `TEMPERATURE` 默认 0（贪心，更稳更快）。

- `TOP_P/TOP_K` 与 `REPETITION_PENALTY/FREQUENCY_PENALTY`：用于抑制小模型复读。

#### vLLM / 平台兼容与防御性处理

- `USE_VLLM`：`auto/true/false`（默认 true）。

- `FORCE_VLLM`：vLLM 初始化失败时是否直接退出（默认回退到 transformers）。

- MetaX 设备探测 `_HAS_MX_DEVICE`：通过 `/dev/mxc*` 判断。

- 插件策略：

  - 非 MetaX 且用户未显式设置 `VLLM_PLUGINS` 时，把它设为空字符串，避免误装插件导致 CUDA 异常。

  - MetaX 上默认启用 `VLLM_USE_V1=1`（平台兼容考虑）。

  - MetaX 上默认 `GPU_MEMORY_UTILIZATION=0.60`（更保守）。

对应函数：

- `_unset_env_if_blank(key)`：修复某些运行时把 `CUDA_VISIBLE_DEVICES` 设为空字符串导致的 vLLM/torch 报错。

- `_has_c_compiler()`：检测 `cc/gcc/clang`，用于决定是否优先 vLLM（Triton 可能需要编译）。

- `_env_flag(name, default)`：统一处理布尔环境变量解析。

#### vLLM speculative decoding（可选加速）

对应函数：

- `_build_speculative_config_from_env(abs_model_dir)`

  - 仅 `ENABLE_SPECULATIVE_DECODING=1` 时启用。

  - 支持 `method=ngram`（无需 draft 模型）或 `method=draft_model`（需要 draft 模型目录）。

  - 重要：启用 speculative 时会提示与 `chunked prefill` 的不兼容，并在后续初始化阶段强制关闭 chunked prefill。

- `_try_print_effective_speculative_config(llm_obj)`：尽力从 vLLM 内部对象回读 speculative 配置，仅用于诊断。

#### vLLM 初始化失败时的自动修复

对应函数：

- `_maybe_parse_estimated_max_len(err)`：从异常链里解析 “estimated maximum model length is …”，用于推断更安全的 `max_model_len`。

- `_maybe_fix_compilation_config(engine_kwargs, err)`：针对不同 vLLM 构建对 `compilation_config` 的类型要求（dict vs JSON 字符串）做一次自动转换重试。

- `_maybe_fix_load_format(engine_kwargs, err)`：当 `load_format` 取值不被当前 vLLM/插件接受（例如 awq）时，移除该参数并回退到 auto。

#### Pydantic 请求/响应模型

- `PredictionRequest`

  - `prompt`: `str` 或 `list[str]`（兼容单条与 batch）。

  - 可选的请求级覆盖参数：`max_new_tokens/temperature/top_p/top_k/repetition_penalty/frequency_penalty`。

- `PredictionResponse`

  - `response`: `str` 或 `list[str]`（与请求形态对应）。

- `SystemPromptRequest`

  - `system_prompt`: `str`。

#### 日志控制

- `_coerce_log_level(level_name, default)`：把字符串等级转为 logging level。

- `_set_logger_level_prefix(prefix, level)`：批量压低 `vllm.*` 日志到 WARNING，减少 batch 场景日志开销。

---

## 2）题型判定与长度分流

### 职责

- 根据题干做启发式分类：短答题 / 代码题 / “强代码信号题” / 长答案题。

- 为不同题型选择不同的 `max_new_tokens`，在**准确率**（避免截断）与**吞吐**（避免短题浪费 token）之间折中。

### 核心函数

#### `_normalize_text(s)`

- 统一把输入 `strip().lower()`，便于关键词匹配。

#### `is_code_question(user_prompt)`

- “保守”判断：只在题目明确要求代码/伪代码/实现时返回 True。

- 内置关键词示例：`代码/伪代码/核心代码/kernel<<< / global void / #include / import / def`。

- 支持 `CODE_QUESTION_KEYWORDS` 追加关键词（逗号分隔）。

- 之所以强调保守：避免 basic 里出现 CUDA 术语就被误判为代码题，导致输出上限变大/吞吐下降。

#### `is_hard_code_question(user_prompt)`

- “更强”代码信号：命中才允许走更长的 `MAX_NEW_TOKENS_CODE_HARD`。

- 关键词更少更严格，默认：`kernel<<< / global void / #include`。

- 支持 `HARD_CODE_QUESTION_KEYWORDS` 追加。

- 使用 `HARD_CODE_MIN_HITS` 做命中次数阈值（并在代码里裁剪到 1\~5），降低误判。

#### `is_long_answer_question(user_prompt)`

- 目标：把 plus/bonus 的“算子类题”分流到更长上限，避免严重截断。

- 默认关键词包含：`算子/spmv/gemm/tensor cores/...`，并支持 `LONG_ANSWER_KEYWORDS` 追加。

- 使用 `LONG_ANSWER_MIN_HITS` 阈值（并裁剪到 1\~5）。

#### `pick_max_new_tokens(user_prompt)`

分流优先级（从严到宽）：

1. 若 `DISABLE_TOKEN_ROUTING`：固定返回 `MAX_NEW_TOKENS`。

2. `is_hard_code_question`：返回 `MAX_NEW_TOKENS_CODE_HARD`（极少数）。

3. `is_long_answer_question`：返回 `MAX_NEW_TOKENS_CODE`（plus/bonus 常见）。

4. `is_code_question`：返回 `MAX_NEW_TOKENS_CODE_SOFT`（“可能需要少量代码”的题）。

5. 默认：返回 `MAX_NEW_TOKENS`（短答）。

---

## 3）prompt 组装与输出后处理

### 职责

- 构造最终输入给模型的 prompt（优先走 Qwen chat template）。

- 在不破坏评测契约的前提下，对输出做“轻量后处理”：

  - 剥离 `<think>`

  - 对短答题裁掉示例/控制句子数

  - 尽量减少无效 token

### 核心函数

#### `build_prompt(user_prompt)`

- 兼容兜底：在没有 tokenizer/chat template 时，拼接 `system + 问题 + 答案` 的纯文本提示。

#### `format_as_chat(tokenizer, user_prompt)`

- 默认策略：构造 `messages=[system,user]`，并调用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`。

- 若 tokenizer 不存在或 apply_chat_template 失败：回退到 `build_prompt`。

性能优化（batch 场景很关键）：

- 当 `FAST_CHAT_TEMPLATE=1`（默认在 batch 模式下启用）且 tokenizer 的 `chat_template` 看起来是 Qwen 常见的 `<|im_start|>...assistant` 结构时：

  - 走**字符串拼接快路径**，绕过 Jinja 模板渲染，显著降低 CPU 开销。

#### `strip_think(text)`

- 目的：评测输出纪律（避免返回推理过程）。

- 但为了防止“模型把最终答案也塞进 think 里”，这里做了安全回退：

  1. 优先删除整个 `<think>...</think>` 块；

  2. 若删除后为空，尝试返回 think 内文本；

  3. 若仍为空，仅移除标签本身。

#### `_postprocess_answer(text, user_prompt)`

仅对“短答模式”（即 `pick_max_new_tokens(user_prompt) == MAX_NEW_TOKENS`）启用，避免误伤长答案/代码题。

包含两类裁剪：

- **裁掉示例扩展**（默认开启 `OUTPUT_TRIM_EXAMPLES=1` 且非代码题）：

  - 检测到 “例如/比如/举例：” 就截断到示例前。

- **限制句子数**（`OUTPUT_MAX_SENTENCES`，默认 6，裁剪到 1\~12）：

  - 按 `。！？；\n` 切句，保留前 N 句。

---

## 4）后端选择与兼容（vLLM / vLLM v1 / transformers）

### 职责

- 可选导入依赖：优先 vLLM（吞吐更高），失败回退 transformers（保证可用）。

- 处理不同平台/不同 vLLM 构建的参数兼容性问题。

- 在 batch 场景下提供更高吞吐路径（vLLM 的离线 `LLM.generate(list_prompts)`）。

### 核心代码块与函数

#### 可选依赖探测

通过 try/except 设置：

- `_vllm_ok`：是否成功导入 vLLM 的 AsyncEngine 相关对象。

- `_vllm_v1_ok`：是否成功导入 vLLM v1 的 `AsyncLLM`。

- `_transformers_ok`：是否成功导入 `torch` 与 `transformers`。

#### `should_use_vllm()`

决策逻辑（简化理解）：

- vLLM 不可用（`_vllm_ok=False`）=> False。

- 若没有 C 编译器且未强制（`FORCE_VLLM=0`）=> False（避免 Triton 初始化失败）。

- `USE_VLLM=true` => True（强制尝试，失败再回退）。

- `USE_VLLM=false` => False。

- 否则（auto）=> 只有在 `torch.cuda.is_available()` 为 True 时才倾向 vLLM。

#### `SERVE_VLLM_ENGINE` 与 `_get_vllm_engine_mode()`

- 用 `SERVE_VLLM_ENGINE` 在 `v0`（AsyncLLMEngine）与 `v1`（AsyncLLM）之间选。

- 这里刻意不复用 `VLLM_USE_V1`，避免和平台/插件已有语义冲突。

#### batch 极限吞吐：`_use_vllm_offline_llm_in_batch_mode()`

- `BATCH_MODE=1` 时默认启用（也可用 `VLLM_BATCH_USE_LLM` 显式控制）。

- 原因：Python 层并发 N 个 `engine.generate(...)` 会创建大量 task + 消费 async generator，有调度开销。

- 离线路线用 vLLM 的 `LLM.generate(list_prompts)` 一次提交整批，更贴近引擎设计初衷。

#### 兼容性修复工具函数

- `_maybe_fix_compilation_config(...)`：处理 `compilation_config` 类型差异。

- `_maybe_fix_load_format(...)`：处理 `load_format` 取值差异（如 awq）。

- `_maybe_parse_estimated_max_len(...)`：从异常里推断建议 max_model_len。

---

## 5）应用生命周期（lifespan：加载/初始化/预热）

### 职责

- 服务启动阶段加载 tokenizer + 后端引擎（vLLM 或 transformers）。

- 做一次预热，降低首请求延迟。

- 把所有运行时对象挂载到 `app.state`，供路由层使用。

### 核心流程：`lifespan(app)`

#### 5.1 选择模型目录（abs_model_dir）

- 若用户未设置 `MODEL_DIR`：

  - 优先使用 `MODEL_DIR` 默认值（`./model/$MODEL_ID`）；

  - 但如果默认路径不存在且 `./merged` 存在，会回退到 `./merged`（本地开发友好）。

- 若设置了 `MODEL_DIR`：直接使用它。

然后打印关键诊断信息：

- `MODEL_DIR`、`BATCH_MODE`、预热参数生效值与数据文件是否存在。

#### 5.2 统一加载 tokenizer（尽量）

- 即使最终走 vLLM，也会尽量先加载 tokenizer：

  - 用于 `format_as_chat`（chat template）与 stop token id 构建。

- tokenizer 失败不会阻塞启动（只影响 prompt 格式优化）。

#### 5.3 初始化 vLLM（若选择 vLLM）

核心策略：

- 使用 `AsyncEngineArgs` 的签名反射（`inspect.signature`）判断参数是否支持，避免不同 vLLM 版本因“多传了一个参数”而启动失败。

- 构造 `engine_kwargs`，其中常见关键项：

  - `model`、`tensor_parallel_size`、`gpu_memory_utilization`、`dtype`、`disable_log_stats`。

可选能力（支持才设置）：

- 量化/加载格式：`VLLM_QUANTIZATION`、`VLLM_LOAD_FORMAT`。

- KV cache dtype：`VLLM_KV_CACHE_DTYPE`/`KV_CACHE_DTYPE`。

- 前缀缓存：`enable_prefix_caching`（默认 True）。

- tokenizer pool：`VLLM_TOKENIZER_POOL_SIZE/TYPE`。

- 容量：`VLLM_MAX_NUM_SEQS`、`VLLM_MAX_NUM_BATCHED_TOKENS`。

- MetaX 上的保守默认：`DEFAULT_MAX_MODEL_LEN`、`ENABLE_CHUNKED_PREFILL` 等。

- speculative decoding：若配置存在且参数支持，注入 `speculative_config`，并强制关闭 chunked prefill。

引擎构建分支：

- 若启用离线 LLM 路线：构建 `vllm.LLM(**llm_kwargs)`，并设置 `app.state.llm`（同步接口）。

- 否则：

  - `SERVE_VLLM_ENGINE=v1` 且可用时构建 `AsyncLLM`（v1）；

  - 否则构建 `AsyncLLMEngine`（v0）。

失败回退策略（关键，保证“能启动”）：

- 离线 LLM 初始化失败：尝试回退到 AsyncLLMEngine。

- V1 引擎失败：回退到 V0 引擎。

- MetaX 特殊回退：

  1. `compilation_config` 自动修复重试；

  2. `load_format` 自动修复重试；

  3. 再失败则尝试 `enforce_eager=True`；

  4. 再失败则尝试降低 `max_model_len`（解析异常推荐值或用 `SAFE_MAX_MODEL_LEN`）。

#### 5.4 vLLM 预热 `_warmup_vllm()`

- 预热时也用 `format_as_chat`，用于真正预热 system prompt 的前缀缓存。

- 构造 `SamplingParams(max_tokens=8)`。

- 如果能从数据文件抽样，则把抽样 prompt 也纳入 warmup（并可 `WARMUP_REPEAT` 重复）。

- 离线 LLM 路线：把预热 `LLM.generate(...)` 放到 `asyncio.to_thread`，避免阻塞事件循环。

- AsyncEngine 路线：消费 async generator 的输出直到结束。

#### 5.5 transformers 初始化（vLLM 不用/失败回退时）

- 确保 `torch/transformers` 可用。

- 加载 `AutoModelForCausalLM`（`device_map="auto"`，`low_cpu_mem_usage=True`）。

- 做一次 `model.generate(max_new_tokens=8)` 预热。

#### 5.6 状态标记

- `app.state.ready = True` 表示服务就绪；health check 会用到。

- 退出时打印 “Shutting down...”。

---

## 6）HTTP 路由（契约与推理主流程）

### 职责

- 提供评测必须的契约：

  - `GET /`：快速返回（judge 会先探活）。

  - `POST /predict`：输入 `{"prompt": "..."}`，返回 `{"response": "..."}`。

- 额外提供 WebUI/调试接口：`/info` 与 `/system_prompt`。

### 核心对象

- `app = FastAPI(title="LLM Service", lifespan=lifespan)`：把生命周期钩子绑定到 FastAPI。

### `GET /`：`health_check()`

- 若未 ready：返回 `{"status":"warming"}`。

- 若 batch 模式：返回 `{"status":"batch"}`（提示评测端一次性推 batch）。

- 否则：返回 `{"status":"ok"}`。

这条路由必须极快：不做任何推理、联网或重计算。

### `GET /info`：`info()`

- 返回后端信息与默认参数，供 WebUI 展示。

- 对环境变量做白名单过滤（`env_keys`），避免泄露敏感信息。

### `GET /system_prompt`：`get_system_prompt_api()`

- 读取当前 system prompt（快速返回）。

### `POST /system_prompt`：`set_system_prompt_api(req)`

- 运行时更新 system prompt，立即影响后续 `/predict` 的 prompt 组装。

### `POST /predict`：`predict(req)`（核心）

整体步骤：

1. **把输入统一成 list**

   - 内部 `to_list()`：将 `str` 或 `list[str]` 统一成 `list[str]`。

2. **prompt 组装**

   - `prompt_texts = [format_as_chat(tokenizer, p) for p in prompts]`。

3. **确定每条样本的 max_new_tokens**

   - 若请求显式传 `req.max_new_tokens`：对 batch 内所有样本用同一个上限（并裁剪到 1\~4096）。

   - 否则：对每条 prompt 用 `pick_max_new_tokens(p)` 路由。

   - 若 `LOG_TOKEN_ROUTING=1`：会统计每个 max_tokens 桶的数量，并用 `uvicorn.error` 打日志。

4. **按后端分支执行**

#### 4A. vLLM 离线 LLM 路线（`app.state.llm` 存在）

- 关键性能点：

  - **不要**给 vLLM 传 “per-prompt 的 SamplingParams 列表”。

  - 这里选择按 `max_tokens` 分桶：同一桶内用同一个 `SamplingParams` 一次性 `LLM.generate(...)`。

  - 这样 batching 才能生效，否则会出现 tokens/s 暴跌。

- 还做了两类缓存：

  - `sampling_params_cache`：按 `(max_tokens, temperature, top_p, ...)` 缓存 SamplingParams。

  - `llm_lock`：通过 `asyncio.Lock` 串行化 LLM.generate（避免并发下内部资源竞争）。

- 输出处理：每条输出先 `strip_think`，再 `_postprocess_answer`（短答才裁剪）。

#### 4B. vLLM AsyncEngine 路线（`app.state.engine`）

- 单条：直接 `run_one()`。

- batch：用 `asyncio.Semaphore(BATCH_CONCURRENCY)` 控制并发，同时触发 vLLM 内部 batching。

- `request_id` 用 `uuid.uuid4().hex` 保证唯一。

- v1/v0 的 generate 调用签名不同：

  - v1: `engine.generate(request_id=..., prompt=..., sampling_params=...)`

  - v0: `engine.generate(prompt, sampling_params, request_id)`

#### 4C. transformers 路线

- 单条：构造 `gen_kwargs` 后 `model.generate()`；再 decode；去掉 prompt 前缀；`strip_think`；`_postprocess_answer`。

- batch：为了兼容协议仍返回 list，但**串行逐条**跑（transformers 的 batch 往往更慢且更吃显存）。

1. **返回形态与输入一致**

   - 输入是 `str` => 返回 `response: str`

   - 输入是 `list[str]` => 返回 `response: list[str]`，且长度必须一致（评测要求）。