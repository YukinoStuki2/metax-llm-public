---
title: 推理服务（参数与环境变量）
sidebar_position: 11
---

本页是对 `serve.py` 的“参数/环境变量/默认值/生效逻辑”的逐项说明。

- 服务启动：见 [推理服务（用法）](./serve)。

- 实现细节：见 [推理服务（代码详解）](./serve_code)。

- 常见问题：见 [推理服务（FAQ）](./serve_faq)。

## 参数入口与优先级

`serve.py` 的“可调参数”来自两类入口：

1. **环境变量（全局默认）**

- 启动进程时读取，影响模型加载/后端选择/默认生成参数。

- 大部分参数在模块 import 时就被读取并固化为全局常量；少数（如 `BATCH_MODE`）刻意做成“运行时动态读取”。

1. **请求级覆盖（仅 /predict）**

- `POST /predict` 的 JSON 里允许带：`max_new_tokens / temperature / top_p / top_k / repetition_penalty / frequency_penalty`。

- 这些字段**只覆盖本次请求**，不改变环境变量与全局默认。

> 注意：评测机只要求 `{"prompt":"..."}`，但本服务为 WebUI/调参保留了请求级覆盖能力。

## 路径与模型来源

### `MODEL_DIR`

- **默认值**：`./model/$MODEL_ID`（其中 `MODEL_ID` 默认是 `YukinoStuki/Qwen3-4B-Plus-LLM`）。

- **作用**：指定要加载的本地 HF 模型目录。

- **细节**：如果环境变量里没有显式设置 `MODEL_DIR`，并且默认目录不存在，但 `./merged` 目录存在，则会自动改用 `./merged` 作为兜底（方便本地 merge 后直接跑）。

### `MODEL_ID`

- **默认值**：`YukinoStuki/Qwen3-4B-Plus-LLM`

- **作用**：用于设置模型modelscope的下载仓库，且拼出模型目录变量 `MODEL_DIR=./model/$MODEL_ID`。

### `TRANSFORMERS_NO_TORCHVISION`

- **默认行为**：代码里 `os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")`

- **作用**：避免 transformers 在启动时引入 torchvision，减少 warning 与潜在版本不匹配问题。

## 生成长度与“按题型分流”

服务默认追求“短答、高 RougeL、少 token”，但会对少量题型（代码题/长答案题）放宽生成长度，避免严重截断。

### `MAX_NEW_TOKENS`

- **默认值**：`32`

- **作用**：短答题的默认上限。

### `MAX_NEW_TOKENS_CODE`

- **默认值**：`192`

- **作用**：对“长答案题”（默认启用，见 `LONG_ANSWER_*`）的上限；也作为一些代码题的上限基准。

### `MAX_NEW_TOKENS_CODE_SOFT`

- **默认值**：`64`

- **作用**：对“可能需要少量代码/伪代码”的题给一个更保守的上限，避免小模型长输出发散。

### `MAX_NEW_TOKENS_CODE_HARD`

- **默认值**：`MAX_NEW_TOKENS_CODE`（也就是默认 192）

- **作用**：对“明确代码形态/核函数”的题允许更长输出（你也可以单独把它调大）。

### `DISABLE_TOKEN_ROUTING`

- **默认值**：`0`

- **作用**：设为`1`时禁用题型分流，所有题都用 `MAX_NEW_TOKENS`。

### `CODE_QUESTION_KEYWORDS`

- **默认值**：空

- **作用**：逗号分隔，追加“代码题”关键词；命中后会走 `MAX_NEW_TOKENS_CODE_SOFT`。

- **注意**：服务内置关键词判定很“严格”，目的是避免把大量非代码题误判为代码题。

### `HARD_CODE_QUESTION_KEYWORDS`

- **默认值**：空

- **作用**：逗号分隔，追加“强代码信号”关键词；命中后会走 `MAX_NEW_TOKENS_CODE_HARD`。

### `HARD_CODE_MIN_HITS`

- **默认值**：`1`（会被裁剪到 1\~5）

- **作用**：强代码关键词需要命中的最小数量。

### `LONG_ANSWER_ENABLE_DEFAULT`

- **默认值**：`1`（真）

- **作用**：是否启用服务内置的一组“长答案题关键词”。

- **内置关键词特点**：偏向区分加分题的“算子类”长题（例如包含“算子/spmv/gemm/triton/tilelang”等）。

### `LONG_ANSWER_KEYWORDS`

- **默认值**：空

- **作用**：逗号分隔，追加“长答案题”关键词。

### `LONG_ANSWER_MIN_HITS`

- **默认值**：`1`（会被裁剪到 1\~5）

- **作用**：长答案关键词需要命中的最小数量。

### `LOG_TOKEN_ROUTING`

- **默认值**：`0`

- **作用**：若开启，会在每次 /predict 输出一个统计日志（不同 `max_new_tokens` 桶的数量），用于观察分流是否符合预期。

## 解码参数（全局默认 + 请求级覆盖）

### `TEMPERATURE`

- **默认值**：`0.0`

- **作用**：0 表示确定性（更稳、也更快）；当请求级覆盖或环境变量使其 > 0 时，transformers 路线会启用采样。

### `TOP_P`

- **默认值**：`1.0`

### `TOP_K`

- **默认值**：`1`

### `REPETITION_PENALTY`

- **默认值**：`1.05`

- **作用**：抑制小模型在贪心解码时的循环复读。

### `FREQUENCY_PENALTY`

- **默认值**：`0.1`

- **作用**：按出现频次惩罚重复。

- **注意**：transformers 路线并非所有版本都支持 `frequency_penalty`；代码里是“尽力透传”。

## Stop（提前停止）

### `STOP_STRINGS`

- **默认值**：空时会自动变为 `"<|im_end|>", "<|endoftext|>"`

- **格式**：逗号分隔字符串列表（会 `strip()`）。

- **作用**：vLLM 路线会作为 `SamplingParams.stop`，transformers 路线会转成 `eos_token_id`（token id 列表）。

### `STOP_ON_DOUBLE_NEWLINE`

- **默认值**：`0`

- **作用**：开启后会把 `"\n\n"` 插到 stop 列表头部。

- **风险**：可能过早截断（但有时能显著减少尾巴 token，提升吞吐）。

## 输出后处理（面向评测）

> 目标：提高 Rouge-L 的词序/短语重合度，减少小模型在“举例扩展”处的发散。

### `OUTPUT_TRIM_EXAMPLES`

- **默认值**：`1`

- **作用**：在短答模式且非代码题时，若输出出现“例如/比如/举例：”，会裁掉其后的扩展内容。

### `OUTPUT_MAX_SENTENCES`

- **默认值**：`6`（会被裁剪到 1\~12）

- **作用**：短答模式下最多保留 N 个“句子片段”（按 `。！？；\n` 进行切分）。

## 预热（Warmup）

> 预热用于降低首请求延迟，但评测机 health 阶段有时间限制，因此代码默认设计为“尽量轻、失败可忽略”。

### `WARMUP_DATA_PATH`

- **默认值**：`./data.jsonl`

- **作用**：可选从本地数据集抽样更多 user prompt 用于预热。

- **注意：这同样也是训练模型的数据集**。

### `WARMUP_NUM_SAMPLES`

- **默认值**：`64`

- **作用**：从数据集抽样的数量。

### `WARMUP_NUM_SAMPLES_CAP`

- **默认值**：`512`（内部会再裁剪到 0\~8192）

- **作用**：给 `WARMUP_NUM_SAMPLES` 加防御性上限，避免误配置导致启动/health 阶段过慢。

### `WARMUP_REPEAT`

- **默认值**：`1`（会被裁剪到 1\~8）

- **作用**：把预热 prompt 列表重复 N 次（适合离线 LLM 批推理路径，把 prefix cache 热起来）。

## Batch 模式与并发

### `BATCH_MODE`

- **默认值**：`0`

- **作用**：开启后：

  - `GET /` 返回 `{"status":"batch"}`（评测机可据此一次性推送所有问题）。

  - `POST /predict` 支持 `prompt` 为 list，返回 `response` 为 list。

- **实现细节**：`is_batch_mode()` 会在运行时读取环境变量，避免“导入时固化”导致 health 与实际模式不一致。

### `BATCH_CONCURRENCY`

- **默认值**：`320`

- **作用**：在 vLLM AsyncEngine 路线下，对 batch 请求用 `asyncio.Semaphore` 限制并发 submit 数，目标是“触发引擎内 batching”同时不把 Python 调度/内存顶爆。

## vLLM / Transformers 后端选择

### `USE_VLLM`

- **默认值**：`true`

- **取值**：`true/false/auto(其他)`

- **作用**：

  - `true`：强制尝试 vLLM（失败可回退，除非 `FORCE_VLLM=1`）。

  - `false`：禁用 vLLM，直接用 transformers。

  - 其他：自动判断（CUDA 可用时倾向 vLLM）。

### `FORCE_VLLM`

- **默认值**：`0`

- **作用**：vLLM 初始化失败时直接抛错退出，不回退到 transformers。

### `VLLM_LOG_LEVEL`

- **默认值**：`WARNING`

- **作用**：降低 vLLM “话痨日志”对吞吐的影响（尤其 batch 模式）。

### `SERVE_VLLM_ENGINE`

- **默认值**：`v0`

- **作用**：选择 vLLM 引擎实现：

  - `v0`：`AsyncLLMEngine`（默认，最稳）

  - `v1`：`AsyncLLM`（可能更快但兼容性风险更高；不可用会自动回退 v0）

### `VLLM_BATCH_USE_LLM`

- **默认值**：`1`（仅在 `BATCH_MODE=1` 时默认启用）

- **作用**：在 batch 模式下改走 vLLM 的同步离线接口 `LLM.generate(list_prompts)`，用一次调用完成整个 batch，减少 Python async 调度开销。

### `FAST_CHAT_TEMPLATE`

- **默认值**：batch 模式默认为真（否则默认假）

- **作用**：对 Qwen 系列常见的 `chat_template` 走字符串拼接快路径，减少 Jinja 渲染 CPU 开销。

### `DEBUG_NET`

- **默认值**：`0`

- **作用**：仅用于调试开关占位。

- **注意**：评测 run 阶段断网，服务端不会在请求路径做联网探测；不要把“联网检查”加到 /predict 里。

## vLLM 关键性能/稳定性参数

这些变量只有在 vLLM 路线下生效，并且代码会做“参数存在才设置”，以兼容不同 vLLM/平台插件版本。

### `GPU_MEMORY_UTILIZATION`

- **默认值**：一般环境默认 `0.85`；MetaX 设备上若未设置则强制更保守的 `0.60`

- **作用**：控制 KV cache 预留/并发容量。

### `DTYPE`

- **默认值**：`float16`

- **作用**：vLLM 模型加载 dtype（字符串形式传入）。

### `MAX_MODEL_LEN`

- **默认值**：不设则由 vLLM/模型配置决定；MetaX 设备上若不设，会用 `DEFAULT_MAX_MODEL_LEN`（默认 38400）兜底。

- **作用**：限制最大序列长度，显著影响 KV cache 占用与并发。

### `DEFAULT_MAX_MODEL_LEN`

- **默认值**：`38400`

- **作用**：仅在 MetaX 设备上、且用户没显式设 `MAX_MODEL_LEN` 时使用。

### `SAFE_MAX_MODEL_LEN`

- **默认值**：`38400`

- **作用**：当 vLLM 初始化报“KV cache 不足”且无法解析出建议长度时，用该值重试一次。

### `VLLM_DEVICE`

- **默认值**：如果 vLLM 构建支持 `device` 参数，则默认 `cuda`

- **额外行为**：若该环境变量存在但为空字符串，会被代码删除（防止云运行时误注入空值导致 vLLM/torch 报错）。

### `VLLM_PLUGINS`

- **默认值**：非 MetaX 机器且未显式设置时，会被设置为空字符串 `""`（禁止加载任何插件）。

- **作用**：避免在非 MetaX 环境误加载 `vllm_metax` 等平台插件导致 CUDA 异常。

### `VLLM_QUANTIZATION`

- **默认值**：空

- **作用**：若当前 vLLM 构建支持 `quantization` 参数，则会透传该值（例如 `awq`）。

### `VLLM_LOAD_FORMAT`

- **默认值**：空

- **作用**：若当前 vLLM 构建支持 `load_format` 参数则透传。

- **兼容性处理**：若出现 `'awq' is not a valid loadFormat` 之类错误，会自动移除 `load_format` 并重试一次。

### `VLLM_KV_CACHE_DTYPE` / `KV_CACHE_DTYPE`

- **默认值**：空

- **作用**：若 vLLM 构建支持 `kv_cache_dtype` 参数，则透传该值以尝试提高 KV cache 容量/并发。

### `ENABLE_PREFIX_CACHING`

- **默认值**：真（若 vLLM 构建支持 `enable_prefix_caching`）

- **作用**：对共享前缀（system prompt）的大 batch 通常能显著加速。

### `VLLM_TOKENIZER_POOL_SIZE` / `VLLM_TOKENIZER_POOL_TYPE`

- **默认值**：空（只有显式设置才启用）

- **作用**：大 batch 场景降低分词 CPU 开销；不同平台支持的 pool type 可能不一致。

### `VLLM_MAX_NUM_SEQS`

- **默认值**：空

- **作用**：若支持 `max_num_seqs` 参数，可显式控制最大并发序列数。

### `VLLM_MAX_NUM_BATCHED_TOKENS`

- **默认值**：空

- **作用**：若支持 `max_num_batched_tokens`，可限制一次批处理的 token 总量。

### `VLLM_ENFORCE_EAGER`

- **默认值**：假

- **作用**：仅在 MetaX 设备上使用；若初始化失败，会自动回退一次 `enforce_eager=True`（更保守但可能更慢）。

### `ENABLE_CHUNKED_PREFILL`

- **默认值**：MetaX 设备上默认假（保守）

- **作用**：仅当 vLLM 构建支持 `enable_chunked_prefill` 参数。

- **注意**：启用 speculative decoding 时会强制关闭（两者当前不兼容）。

### `VLLM_COMPILATION_CONFIG`

- **默认值**：空

- **作用**：若 vLLM 构建支持 `compilation_config` 参数，则把该变量（建议为 JSON 字符串）透传。

- **兼容性处理**：若遇到常见 Pydantic 类型错误，代码会尝试在 dict/JSON 字符串之间自动转换一次后重试。

## Speculative Decoding（可选）

> 默认关闭；实测会拖慢小模型速度，沐曦vllm插件0.11.0不支持draft推理解码。

### `ENABLE_SPECULATIVE_DECODING`

- **默认值**：假

- **作用**：开启后，会构造 `speculative_config` 并注入到 vLLM 引擎参数中（若当前构建支持）。

### `SPEC_NUM_SPECULATIVE_TOKENS`

- **默认值**：`6`（裁剪到 1\~32）

### `SPEC_METHOD`

- **默认值**：`draft_model`

- **可选**：`ngram` 或其他 vLLM 支持的方法。

### `SPEC_NGRAM_LOOKUP_MAX` / `SPEC_NGRAM_LOOKUP_MIN`

- **仅在** `SPEC_METHOD=ngram` 时生效

- **默认值**：max=8、min=1

### `SPEC_DRAFT_MODEL_DIR`

- **默认值**：空

- **作用**：指定 draft 模型本地目录。

### `SPEC_DRAFT_MODEL_ID`

- **默认值**：空

- **作用**：若未提供 `SPEC_DRAFT_MODEL_DIR`，则可用该 ID 自动拼出 `./model/$SPEC_DRAFT_MODEL_ID`。

### `SPEC_DRAFT_MAX_MODEL_LEN`

- **默认值**：空

- **作用**：可显式设置 draft 模型的 `max_model_len`。

### `SPEC_DISABLE_BY_BATCH_SIZE`

- **默认值**：空

- **作用**：当 batch 排队过大时自动禁用 speculation（避免二模型拖慢）。

## Transformers 后端参数

### `TRANSFORMERS_DTYPE`

- **默认值**：`float16`

- **作用**：transformers `AutoModelForCausalLM.from_pretrained(..., dtype=...)` 的 dtype 映射。

## 运行时/平台相关变量（了解即可）

这些变量要么是通用运行时变量，要么是 `serve.py` 为兼容性主动设置的变量；通常不需要手动调。

### `VLLM_USE_MODELSCOPE`

- **默认行为**：`serve.py` 会设置为 `True`（字符串）

- **作用**：保留给部分 vLLM 构建/环境在加载模型时使用 ModelScope 的兼容路径。

### `VLLM_USE_V1`

- **默认行为**：在检测到 MetaX 设备时，如果用户未显式设置，会设为 `1`

- **作用**：可能被平台插件用作 vLLM 内部行为开关。

- **注意**：服务端选择 vLLM 引擎实现使用的是 `SERVE_VLLM_ENGINE`，不是 `VLLM_USE_V1`。

### `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`

- **作用**：控制容器/进程可见的 GPU 设备。

- **服务端额外处理**：若变量存在但值为空字符串，会被删除（避免 vLLM/torch 报“设备字符串不能为空”）。