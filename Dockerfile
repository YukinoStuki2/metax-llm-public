FROM cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64

WORKDIR /app
ENV OMP_NUM_THREADS=4
ENV PATH="/opt/conda/bin:$PATH"

COPY requirements.txt .
COPY download_model.py .

RUN pip install --no-cache-dir -r requirements.txt

# 下载你上传到 ModelScope 的 merged 权重
RUN python download_model.py \
        --model_name yukinostuki/qwen3-4b-ft-v1 \
        --cache_dir /app/model \
        --revision master

# 对齐 snapshot_download 的真实目录：/app/model/<namespace>/<repo>
ENV MODEL_DIR=/app/model/yukinostuki/qwen3-4b-ft-v1

# 强烈建议：评测环境直接用 vLLM
ENV USE_VLLM=true
ENV MAX_NEW_TOKENS=48

EXPOSE 8000

COPY . .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
