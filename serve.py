import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ======================
# 配置区：容器优先
# ======================
# Dockerfile 里会 ENV MODEL_DIR=/app/model/yukinostuki/qwen3-4b-ft-v1
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model/yukinostuki/qwen3-4b-ft-v1")

# 评测速度很关键：默认输出短一点
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "48"))

# TEMPERATURE=0 -> 确定性生成（更稳更快）
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TOP_K = int(os.environ.get("TOP_K", "50"))

# vLLM 开关：auto/true/false
USE_VLLM = os.environ.get("USE_VLLM", "true").lower()

# ⚠️ 评测 run 阶段断网：不要每次 predict 进行联网探测（会拖慢）
DEBUG_NET = os.environ.get("DEBUG_NET", "0") == "1"

# vLLM 在部分环境可用 ModelScope（保留）
os.environ["VLLM_USE_MODELSCOPE"] = "True"

_request_id = 1


def strip_think(text: str) -> str:
    """去掉可能的 <think>...</think>，避免输出过长。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class PredictionRequest(BaseModel):
    prompt: str


class PredictionResponse(BaseModel):
    response: str


# ======================
# 可选依赖探测
# ======================
_vllm_ok = False
_transformers_ok = False

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    _vllm_ok = True
except Exception:
    _vllm_ok = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _transformers_ok = True
except Exception:
    _transformers_ok = False


def should_use_vllm() -> bool:
    if USE_VLLM == "true":
        return True
    if USE_VLLM == "false":
        return False
    return _vllm_ok


def build_prompt(user_prompt: str) -> str:
    # 评测很吃速度：尽量短输出、不要思维链
    prefix = "请用中文一句话简洁作答，不要解释推理过程。\n"
    return prefix + user_prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型（评测 health check 阶段也会触发）。"""
    abs_model_dir = os.path.abspath(MODEL_DIR)
    print("MODEL_DIR =", abs_model_dir)

    if not os.path.isdir(abs_model_dir):
        raise RuntimeError(f"MODEL_DIR not found: {abs_model_dir}")

    # 先选后端：vLLM（若可用）否则 transformers
    if should_use_vllm():
        print("Initializing vLLM engine...")
        try:
            engine_args = AsyncEngineArgs(
                model=abs_model_dir,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.60,
                trust_remote_code=True,
            )
            app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
            app.state.backend = "vllm"
            print("vLLM engine initialized successfully!")
        except Exception as e:
            print("vLLM init failed, fallback to transformers. Error:", e)
            app.state.backend = "transformers"
    else:
        app.state.backend = "transformers"

    if app.state.backend == "transformers":
        if not _transformers_ok:
            raise RuntimeError("Transformers backend selected but torch/transformers not installed.")

        print("Initializing Transformers model...")
        tok_kwargs = dict(trust_remote_code=True)
        # 兼容：某些 tokenizer 会提示 regex，尽量修复（不支持也无妨）
        tok_kwargs["fix_mistral_regex"] = True

        tokenizer = AutoTokenizer.from_pretrained(abs_model_dir, **tok_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            abs_model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # 预热（减少首请求延迟）
        try:
            _ = model.generate(
                **tokenizer("你好", return_tensors="pt").to(model.device),
                max_new_tokens=8,
                do_sample=False,
            )
        except Exception as e:
            print("Warmup failed (can ignore):", e)

        app.state.tokenizer = tokenizer
        app.state.model = model
        print("Transformers model initialized successfully!")

    yield
    print("Shutting down...")


app = FastAPI(title="LLM Service", lifespan=lifespan)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    global _request_id

    prompt_text = build_prompt(req.prompt)
    backend = getattr(app.state, "backend", "transformers")

    if backend == "vllm":
        engine = app.state.engine
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_NEW_TOKENS,
        )

        results = engine.generate(prompt_text, sampling_params, str(_request_id))
        _request_id += 1

        final_output = None
        async for request_output in results:
            final_output = request_output

        text = final_output.outputs[0].text.strip()
        return PredictionResponse(response=strip_think(text))

    # transformers 路线
    tok = app.state.tokenizer
    mdl = app.state.model

    inputs = tok(prompt_text, return_tensors="pt").to(mdl.device)

    do_sample = TEMPERATURE > 0
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K))

    out = mdl.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0], skip_special_tokens=True)

    if text.startswith(prompt_text):
        text = text[len(prompt_text):].strip()

    return PredictionResponse(response=strip_think(text))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
