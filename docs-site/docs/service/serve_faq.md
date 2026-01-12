---
title: 推理服务（FAQ）
sidebar_position: 13
---

本页整理 `serve.py` 在评测/本地复现/云机器环境里最常见的问题与处理建议。

- 用法与定位：看 [推理服务（用法与定位）](./serve)
- 参数与环境变量：看 [推理服务（参数与环境变量）](./serve_env)
- 实现细节：看 [推理服务（代码详解）](./serve_code)

## 1) 健康检查一直是 warming / 超时

**现象**：`GET /` 长时间返回 `{"status":"warming"}`，或评测机 health 阶段超时。

**原因**：启动阶段卡在模型加载/vLLM 初始化/预热上。

**排查与处理**：

- 确认 `MODEL_DIR` 目录存在且包含可加载的模型文件（最少应有 `config.json`、权重文件与 tokenizer 相关文件）。
- 优先减少 warmup 开销：
  - `WARMUP_NUM_SAMPLES=0`（禁用数据集抽样预热）
  - `WARMUP_REPEAT=1`
- 如果 vLLM 初始化很慢或失败：
  - 先让服务能跑起来：`USE_VLLM=false` 强制走 transformers 验证功能链路。
  - 或保留 vLLM，但允许回退：不要设 `FORCE_VLLM=1`。

## 2) 开了 batch 模式但 health 不是 batch

**现象**：你以为设置了 `BATCH_MODE=1`，但 `GET /` 仍返回 `{"status":"ok"}`。

**结论**：服务端 `GET /` 的 batch 判定是运行时读取 `BATCH_MODE`。

**处理**：

- 确认启动时环境变量确实传进进程（容器里可 `printenv | grep BATCH_MODE`）。
- 注意一些脚本可能覆盖环境变量：本项目约定是 `run_model.sh` “有就用、没有才给默认”，而 `env_force.sh` 会强制导入整套默认。

## 3) vLLM 初始化失败，自动回退了 transformers

**现象**：启动日志出现 `vLLM init failed, fallback to transformers`。

**常见原因**：

- 系统缺少 C 编译器（Triton 需要）：`cc/gcc/clang` 不存在。
- CUDA 不可用或驱动不匹配。
- MetaX/平台插件对某些参数不兼容（`load_format/compilation_config` 等）。

**处理建议**：

- 如果你就是要 vLLM：
  - 先确认工具链齐全（至少有 `gcc` 或 `clang`）。
  - 确认 `torch.cuda.is_available()` 为真。
  - 必要时 `FORCE_VLLM=1` 让问题尽早暴露（不回退）。
- 如果只是为了评测可用：允许回退即可（默认行为）。

## 4) 看到 “Device string must not be empty” 之类错误

**原因**：某些云运行时会把 `CUDA_VISIBLE_DEVICES` 或 `VLLM_DEVICE` 注入为空字符串。

**服务端处理**：启动时会把空字符串的设备变量删除（防御性修复）。

**仍然失败时**：

- 检查是否有其他设备相关变量被注入为空：`NVIDIA_VISIBLE_DEVICES`。
- 显式设置 `VLLM_DEVICE=cuda` 或清理相关环境变量再启动。

## 5) OOM、并发很低、或者 tokens/s 很差

### 5.1 OOM（KV cache 或权重加载）

优先顺序：

1) 降低 `MAX_MODEL_LEN`（或在 MetaX 上调低 `DEFAULT_MAX_MODEL_LEN`）
2) 降低 `GPU_MEMORY_UTILIZATION`
3) 限制 vLLM 容量参数：`VLLM_MAX_NUM_SEQS` / `VLLM_MAX_NUM_BATCHED_TOKENS`

### 5.2 tokens/s 很差（尤其 batch）

最常见原因是：batch 路径没有充分 batching。

- 在 vLLM 离线 LLM 路线下，`serve.py` 已经做了关键优化：按 `max_tokens` 分桶批推理。
- 若你自行改动了 /predict，确保不要把 per-prompt `SamplingParams` 列表传给 vLLM，否则容易导致 batching 失效。

另外：

- 开启 `FAST_CHAT_TEMPLATE=1`（batch 模式默认就会开启）可以显著降低 chat template CPU 开销。
- 适当调大 `BATCH_CONCURRENCY` 可能有助于喂饱引擎，但过大也可能让 Python 调度/内存顶爆。

## 6) 输出被截断 / Rouge 掉分

**可能原因**：

- `MAX_NEW_TOKENS` 太小；或分流没有命中，导致长题仍按短答上限生成。

**处理**：

- 如果是代码题：
  - 适当调大 `MAX_NEW_TOKENS_CODE_SOFT` / `MAX_NEW_TOKENS_CODE_HARD`
  - 或追加 `CODE_QUESTION_KEYWORDS` / `HARD_CODE_QUESTION_KEYWORDS`
- 如果是算子类长题：
  - 确认 `LONG_ANSWER_ENABLE_DEFAULT=1`
  - 或追加 `LONG_ANSWER_KEYWORDS`
- 想快速验证分流情况：开启 `LOG_TOKEN_ROUTING=1` 看每次请求的桶统计。

## 7) 输出里有 `<think>...</think>` 或者被清空了

服务端会调用 `strip_think`，其设计目标是：

- 尽量去掉思考过程，避免输出过长。
- 但如果模型把最终答案写在 think 里，也会尝试回退提取。

如果你发现答案仍然异常：

- 先确认模型本身是否总把答案放在 `<think>` 里；这种模型通常不适合此评测设定。
- 可以临时关闭系统提示词中的“不要思考过程”相关描述做对比（通过 `/system_prompt` 更新），但这往往会增加输出长度并拖慢吞吐。

## 8) STOP 过早截断 / 或者 stop 不生效

- `STOP_ON_DOUBLE_NEWLINE=1` 可能导致过早截断（尤其是答案本身需要空行时）。先关掉再观察。
- `STOP_STRINGS` 需要与模型实际输出的结束标记匹配；默认是 Qwen 常见的 `<|im_end|>`、`<|endoftext|>`。
- transformers 路线只使用 stop token ids（`eos_token_id`），不支持任意字符串 stop；因此字符串 stop 更依赖 vLLM 路线。

## 9) WebUI 动态更新 system prompt 没生效

- 更新接口是 `POST /system_prompt`。
- `serve.py` 用全局变量保存当前 prompt，后续 `format_as_chat` 会读取它；不需要重启。
- 如果你自己写了反向代理或缓存层，确认没有缓存住旧结果。

## 10) 为什么还有 /info 和 /system_prompt 这些接口？会影响评测吗？

- 评测只依赖 `GET /` 与 `POST /predict`。
- 额外接口主要服务 WebUI 与调试。
- `/info` 对环境变量做了白名单过滤，避免把 token/密钥暴露出来。

