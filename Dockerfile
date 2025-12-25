FROM cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64

WORKDIR /app
ENV OMP_NUM_THREADS=4
ENV PATH="/opt/conda/bin:$PATH"

COPY requirements.txt .
COPY download_model.py .
COPY merge_adapter.py .

RUN pip install --no-cache-dir -r requirements.txt

# 通过环境变量配置 adapter 仓库地址（默认走 https，避免 docker build 时缺少 ssh key）
ENV ADAPTER_REPO_URL=https://gitee.com/yukinostuki/qwen3-4b-plus.git
ENV BASE_MODEL=Qwen/Qwen3-4B

# 下载基座模型 + clone adapter 仓库并融合成完整权重
RUN python merge_adapter.py \
        --base_model "$BASE_MODEL" \
        --cache_dir /app/model \
        --output_dir /app/model/merged

# 运行时默认加载融合后的目录
ENV MODEL_DIR=/app/model/merged

# 强烈建议：评测环境直接用 vLLM
ENV USE_VLLM=true
ENV MAX_NEW_TOKENS=48

EXPOSE 8000

COPY . .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
