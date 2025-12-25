import os
import re
from contextlib import asynccontextmanager
import asyncio
from typing import Any, List, Optional, Union

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

# Batch 模式：GET / 返回 {"status":"batch"} 后，评测机会一次性把所有问题推到 /predict
# 默认关闭；开启后 /predict 会兼容 prompt 为 list[str]
BATCH_MODE = os.environ.get("BATCH_MODE", "0") == "1"

# batch 并发：并发提交给 vLLM 以触发引擎内 batching
BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "16"))

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
    # 单条：str；batch：list[str]
    prompt: Union[str, List[str]]


class PredictionResponse(BaseModel):
    # 单条：str；batch：list[str]
    response: Union[str, List[str]]


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
    # 兼容：旧逻辑（不会走 chat template 的情况下）
    prefix = "请用中文简洁作答，不要解释推理过程，不要复述问题。\n"
    return prefix + user_prompt


def format_as_chat(tokenizer: Any, user_prompt: str) -> str:
    """用 Qwen 系列的 chat template 构造最终 prompt（提升准确率/一致性）。"""
    p = (user_prompt or "").strip()
    messages = [
        {
            "role": "system",
            "content": "你是中文知识问答助手。请直接给出答案，不要输出推理过程，不要复述问题，答案尽量简短。",
        },
        {"role": "user", "content": p},
    ]

    if tokenizer is None:
        return build_prompt(p)

    try:
        # Qwen/Qwen2/Qwen3 通常支持 apply_chat_template
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # 兜底：仍然可用纯文本提示
        return build_prompt(p)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型（评测 health check 阶段也会触发）。"""
    abs_model_dir = os.path.abspath(MODEL_DIR)
    print("MODEL_DIR =", abs_model_dir)

    if not os.path.isdir(abs_model_dir):
        raise RuntimeError(f"MODEL_DIR not found: {abs_model_dir}")

    # tokenizer：无论后端如何都尽量加载，用于 chat template
    tokenizer = None
    if _transformers_ok:
        try:
            tok_kwargs = dict(trust_remote_code=True)
            tok_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(abs_model_dir, **tok_kwargs)
            app.state.tokenizer = tokenizer
            print("Tokenizer initialized successfully!")
        except Exception as e:
            app.state.tokenizer = None
            print("Tokenizer init failed (can ignore):", e)
    else:
        app.state.tokenizer = None

    # 先选后端：vLLM（若可用）否则 transformers
    if should_use_vllm():
        print("Initializing vLLM engine...")
        try:
            # 注意：尽量只使用通用/稳定参数，避免不同 vLLM 版本不兼容
            engine_args = AsyncEngineArgs(
                model=abs_model_dir,
                tensor_parallel_size=1,
                gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")),
                trust_remote_code=True,
                dtype=os.environ.get("DTYPE", "float16"),
                disable_log_stats=True,
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
        # 如果前面 tokenizer 已加载，直接复用
        tokenizer = getattr(app.state, "tokenizer", None)
        if tokenizer is None:
            tok_kwargs = dict(trust_remote_code=True)
            tok_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(abs_model_dir, **tok_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            abs_model_dir,
            trust_remote_code=True,
            torch_dtype={
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(os.environ.get("TRANSFORMERS_DTYPE", "float16").lower(), torch.float16),
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
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
    if BATCH_MODE:
        return {"status": "batch"}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    global _request_id

    tokenizer = getattr(app.state, "tokenizer", None)

    def to_list(p: Union[str, List[str]]) -> List[str]:
        if isinstance(p, list):
            return [str(x) for x in p]
        return [str(p)]

    prompts = to_list(req.prompt)
    prompt_texts = [format_as_chat(tokenizer, p) for p in prompts]
    backend = getattr(app.state, "backend", "transformers")

    if backend == "vllm":
        engine = app.state.engine
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=1.0 if TEMPERATURE == 0.0 else TOP_P,
            top_k=-1 if TEMPERATURE == 0.0 else TOP_K,
            max_tokens=MAX_NEW_TOKENS,
        )

        async def run_one(text_prompt: str) -> str:
            global _request_id
            rid = str(_request_id)
            _request_id += 1
            results = engine.generate(text_prompt, sampling_params, rid)
            final_output = None
            async for request_output in results:
                final_output = request_output
            if final_output is None or not final_output.outputs:
                return ""
            return strip_think(final_output.outputs[0].text)

        # 单条：直接跑；batch：限制并发以触发 vLLM 内部 batching
        if len(prompt_texts) == 1:
            return PredictionResponse(response=await run_one(prompt_texts[0]))

        sem = asyncio.Semaphore(max(1, BATCH_CONCURRENCY))

        async def guarded(tp: str) -> str:
            async with sem:
                return await run_one(tp)

        outputs = await asyncio.gather(*[guarded(tp) for tp in prompt_texts])
        return PredictionResponse(response=outputs)

    # transformers 路线
    tok = app.state.tokenizer
    mdl = app.state.model

    if len(prompt_texts) != 1:
        # transformers 路线不建议 batch（慢且占显存），但为了兼容协议仍给出串行结果
        outputs: List[str] = []
        for pt in prompt_texts:
            inputs = tok(pt, return_tensors="pt").to(mdl.device)
            do_sample = TEMPERATURE > 0
            gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample)
            if do_sample:
                gen_kwargs.update(dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K))
            out = mdl.generate(**inputs, **gen_kwargs)
            text = tok.decode(out[0], skip_special_tokens=True)
            if text.startswith(pt):
                text = text[len(pt):].strip()
            outputs.append(strip_think(text))
        return PredictionResponse(response=outputs)

    prompt_text = prompt_texts[0]
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
