FROM cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64

WORKDIR /app
ENV OMP_NUM_THREADS=4
ENV PATH="/opt/conda/bin:$PATH"

COPY requirements.txt .
COPY download_model.py .

RUN pip install --no-cache-dir -r requirements.txt

# 直接从 ModelScope 下载已融合的模型（线上运行环境不再执行本地融合）
ENV MODEL_ID=YukinoStuki/Qwen3-4B-Plus-LLM
ENV MODEL_REVISION=master

# Speculative Decoding（可选）：draft 模型
# - 默认不启用（ENABLE_SPECULATIVE_DECODING=0），避免因为不兼容/额外开销影响稳定性。
# - 如需启用：
#   1) 设置 SPEC_DRAFT_MODEL_ID（会在 build 阶段下载到 ./model/$SPEC_DRAFT_MODEL_ID）
#   2) 运行时设置 ENABLE_SPECULATIVE_DECODING=1
ENV ENABLE_SPECULATIVE_DECODING=1 \
        SPEC_DRAFT_MODEL_ID='Qwen3-1.7B-Plus-LLM' \
        SPEC_DRAFT_MODEL_REVISION=master \
        SPEC_NUM_SPECULATIVE_TOKENS=8 \
        SPEC_METHOD=draft_model

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
ENV MAX_NEW_TOKENS=32
ENV MAX_NEW_TOKENS_CODE=192

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
        VLLM_LOAD_FORMAT=auto \
        VLLM_MAX_NUM_SEQS= \
        VLLM_MAX_NUM_BATCHED_TOKENS= \
        VLLM_COMPILATION_CONFIG= \
        MAX_MODEL_LEN=2048 \
        DEBUG_NET=0

# download_model.py 的下载 token（可选；为空表示匿名下载）
ENV MODELSCOPE_API_TOKEN=ms-b0fc501a-2521-4fdd-b54a-fbe6674df836

EXPOSE 8000

COPY . .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
