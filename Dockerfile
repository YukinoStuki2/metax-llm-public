FROM cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64

WORKDIR /app
ENV OMP_NUM_THREADS=4
ENV PATH="/opt/conda/bin:$PATH"

COPY requirements.txt .
COPY download_model.py .

RUN pip install --no-cache-dir -r requirements.txt

# 直接从 ModelScope 下载已融合的模型（线上运行环境不再执行本地融合）
ENV MODEL_ID=YukinoStuki/Qwen2.5-0.5B-Plus-EN
ENV MODEL_REVISION=master

# Speculative Decoding（可选）
# - 默认不启用（ENABLE_SPECULATIVE_DECODING=0），避免因为不兼容/额外开销影响稳定性。
# - ngram 方法：不需要 draft 模型，基于 prompt 内 n-gram 模式预测，零额外成本
# - 如需启用 ngram：运行时设置 ENABLE_SPECULATIVE_DECODING=1
ENV ENABLE_SPECULATIVE_DECODING=0 \
        SPEC_METHOD=ngram \
        SPEC_NUM_SPECULATIVE_TOKENS=6 \
        SPEC_NGRAM_LOOKUP_MAX=8 \
        SPEC_NGRAM_LOOKUP_MIN=1 \
        SPEC_DRAFT_MODEL_ID= \
        SPEC_DRAFT_MODEL_REVISION=master

RUN python download_model.py \
        --model_name "$MODEL_ID" \
        --cache_dir ./model \
                --revision "$MODEL_REVISION" \
        --token "$MODELSCOPE_API_TOKEN" \
                --draft_model_name "$SPEC_DRAFT_MODEL_ID" \
                --draft_revision "$SPEC_DRAFT_MODEL_REVISION" \
                --draft_optional

# 运行时默认加载下载的融合模型目录
ENV MODEL_DIR=./model/$MODEL_ID

# 强烈建议：评测环境直接用 vLLM
ENV USE_VLLM=true
ENV DISABLE_TOKEN_ROUTING=0
ENV MAX_NEW_TOKENS=64
ENV MAX_NEW_TOKENS_CODE=192
ENV MAX_NEW_TOKENS_CODE_HARD=192
ENV MAX_NEW_TOKENS_CODE_SOFT=96
ENV HARD_CODE_MIN_HITS=1
ENV LONG_ANSWER_ENABLE_DEFAULT=1
ENV LONG_ANSWER_MIN_HITS=1
# 解码稳定性：抑制复读（对小模型尤其重要）
ENV REPETITION_PENALTY=1.05
ENV FREQUENCY_PENALTY=0.1

# 输出后处理：非代码题裁剪示例段、限制句子数（提高 Rouge 稳定性）
ENV OUTPUT_TRIM_EXAMPLES=1
ENV OUTPUT_MAX_SENTENCES=6

# vLLM 停止条件（减少无效尾巴 token；可在运行时覆盖）
ENV STOP_STRINGS="<|im_end|>,<|endoftext|>" \
        STOP_ON_DOUBLE_NEWLINE=0

# serve.py 运行时参数（显式写出默认值，避免环境不一致）
ENV BATCH_MODE=1 \
        BATCH_CONCURRENCY=512 \
        TEMPERATURE=0.0 \
        TOP_P=1.0 \
        TOP_K=1 \
        GPU_MEMORY_UTILIZATION=0.97 \
        DTYPE=float16 \
        TRANSFORMERS_DTYPE=float16 \
        ENABLE_PREFIX_CACHING=1 \
        VLLM_QUANTIZATION= \
        VLLM_KV_CACHE_DTYPE= \
        VLLM_LOAD_FORMAT=auto \
        VLLM_MAX_NUM_BATCHED_TOKENS= \
        VLLM_COMPILATION_CONFIG= \
        VLLM_MAX_NUM_SEQS=1024 \
        MAX_MODEL_LEN=1024 \
        WARMUP_DATA_PATH=./data.jsonl \
        WARMUP_NUM_SAMPLES=512 \
        WARMUP_NUM_SAMPLES_CAP=512 \
        WARMUP_REPEAT=2 \
        DEBUG_NET=0

# download_model.py 的下载 token（可选；为空表示匿名下载）
ENV MODELSCOPE_API_TOKEN=

EXPOSE 8000

COPY . .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
