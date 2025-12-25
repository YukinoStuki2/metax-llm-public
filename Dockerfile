FROM cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64

WORKDIR /app
ENV OMP_NUM_THREADS=4
ENV PATH="/opt/conda/bin:$PATH"

COPY requirements.txt .
COPY download_model.py .

RUN pip install --no-cache-dir -r requirements.txt

# 直接从 ModelScope 下载已融合的模型（线上运行环境不再执行本地融合）
ENV MODEL_ID=yukinostuki/qwen3-4b-ft-v1
ENV MODEL_REVISION=latest

RUN python download_model.py \
        --model_name "$MODEL_ID" \
        --cache_dir ./model \
        --revision "$MODEL_REVISION"

# 运行时默认加载下载的融合模型目录
ENV MODEL_DIR=./model/$MODEL_ID

# 强烈建议：评测环境直接用 vLLM
ENV USE_VLLM=true
ENV MAX_NEW_TOKENS=48

# serve.py runtime knobs (keep defaults explicit)
ENV BATCH_MODE=0 \
        BATCH_CONCURRENCY=16 \
        TEMPERATURE=0.0 \
        TOP_P=1.0 \
        TOP_K=1 \
        GPU_MEMORY_UTILIZATION=0.85 \
        DTYPE=float16 \
        TRANSFORMERS_DTYPE=float16 \
        DEBUG_NET=0

# download_model.py (optional; empty means anonymous download)
ENV MODELSCOPE_API_TOKEN=

EXPOSE 8000

COPY . .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
