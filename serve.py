import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uvicorn
import socket

def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


class PredictionRequest(BaseModel):
    prompt: str

class PredictionResponse(BaseModel):
    response: str

# 设置环境变量，启用 vLLM 的 ModelScope 支持
os.environ['VLLM_USE_MODELSCOPE']='True'

# 全局计数器
count = 1

# 模型本地路径映射字典
model_local_dict = {
    "Qwen/Qwen3-1.7B": "/app/Qwen/Qwen3-1.7B"
}

@asynccontextmanager
async def initializationEngine(app: FastAPI):
    '''
        初始化引擎
    '''
    print("Initializing vLLM engine...")
    try:
        engine_args = AsyncEngineArgs(
            model=model_local_dict["Qwen/Qwen3-1.7B"],
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            trust_remote_code=True
        )
        app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("vLLM engine initialized successfully!")
    except Exception as e:
        print(f"Engine initialization failed: {e}")
        raise
    yield
    print("Shutting down vLLM engine...")

app = FastAPI(title="vLLM Service", lifespan=initializationEngine)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    文本生成预测接口
    Example:
        >>> 请求
        {"prompt": "今天天气很好，"}
        
        >>> 响应  
        {"response": "适合出去散步。"}
        
    Notes:
        - 使用全局计数器(count)确保每次生成使用不同的请求ID
        - 生成的文本会去除首尾空白字符
    """
    global count
    engine = app.state.engine
    prompt_text = request.prompt

    raise RuntimeError(request.prompt)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=50,
    )
    results_generator = engine.generate(prompt_text, sampling_params, str(count))
    count += 1
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    generated_text = final_output.outputs[0].text

    # --- 网络连通性测试 ---
    internet_ok = check_internet()
    print("【Internet Connectivity Test】:",
        "CONNECTED" if internet_ok else "OFFLINE / BLOCKED")

    return PredictionResponse(response=generated_text.strip())

@app.get("/")
def health_check():
    '''
        在health_check阶段会执行 `initializationEngine` 函数，
        Timeout: 180s
    '''
    return {"status": "ok"}