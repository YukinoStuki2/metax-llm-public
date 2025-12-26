import os
import re
from contextlib import asynccontextmanager
import asyncio
import uuid
from typing import Any, List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ======================
# 配置区：容器优先
# ======================
# Dockerfile / run_model.sh 下载到 ./model/$MODEL_ID；默认使用同一路径
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join("./model", os.environ.get("MODEL_ID", "YukinoStuki/Qwen3-4B-Plus-Merged")),
)

# 生成参数（评测要求）
SYSTEM_PROMPT = (
    "你是评测答题模型。目标：ROUGE-L高分且尽量少输出token。\n"
    "只输出答案正文，不要任何“思考过程/推理/分析/步骤/解释/客套”，不要出现“思考完成”等字样。\n\n"
    "写法要求：\n"
    "1) 尽量复用教材/标准表述，少改写，保持常见措辞与词序。\n"
    "2) 用3-6个短句/短语覆盖关键点（定义/参数/公式/步骤/关键术语），不要长段落。\n"
    "3) 不举例、不扩展背景、不重复。\n"
    "4) 若题目要求代码：只输出最短可用的核心代码/伪代码骨架，不加Markdown围栏，不解释。"
)

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
WARMUP_PROMPT = "你好"

# Batch 模式：GET / 返回 {"status":"batch"} 后，评测机会一次性把所有问题推到 /predict
# 默认关闭；开启后 /predict 会兼容 prompt 为 list[str]
BATCH_MODE = os.environ.get("BATCH_MODE", "0") == "1"

# batch 并发：并发提交给 vLLM 以触发引擎内 batching
BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "16"))

# TEMPERATURE=0 -> 确定性生成（更稳更快）
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TOP_K = int(os.environ.get("TOP_K", "1"))

# vLLM 开关：auto/true/false
USE_VLLM = os.environ.get("USE_VLLM", "true").lower()

# ⚠️ 评测 run 阶段断网：不要每次 predict 进行联网探测（会拖慢）
DEBUG_NET = os.environ.get("DEBUG_NET", "0") == "1"

# vLLM 在部分环境可用 ModelScope（保留）
os.environ["VLLM_USE_MODELSCOPE"] = "True"


def _unset_env_if_blank(key: str) -> None:
    """Unset env var if it exists but is blank.

    Some cloud runtimes accidentally set CUDA_VISIBLE_DEVICES="" which can
    trigger vLLM/torch errors like: "Device string must not be empty".
    """
    if key in os.environ and os.environ.get(key, "").strip() == "":
        del os.environ[key]


# Defensive: avoid empty device strings breaking vLLM.
_unset_env_if_blank("CUDA_VISIBLE_DEVICES")
_unset_env_if_blank("NVIDIA_VISIBLE_DEVICES")
_unset_env_if_blank("VLLM_DEVICE")

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
    if not _vllm_ok:
        return False

    # Prefer vLLM only when CUDA is actually available.
    cuda_ok = False
    try:
        import torch  # type: ignore

        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if USE_VLLM == "true":
        return cuda_ok
    if USE_VLLM == "false":
        return False
    return cuda_ok


def build_prompt(user_prompt: str) -> str:
    # 兼容：旧逻辑（不会走 chat template 的情况下）
    return f"{SYSTEM_PROMPT}\n问题：{user_prompt}\n答案："


def format_as_chat(tokenizer: Any, user_prompt: str) -> str:
    """用 Qwen 系列的 chat template 构造最终 prompt（提升准确率/一致性）。"""
    p = (user_prompt or "").strip()
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
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

    app.state.ready = False

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

            async def _warmup_vllm():
                try:
                    sp = SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=TOP_K,
                        max_tokens=8,
                        frequency_penalty=0.0,
                    )
                    gen = app.state.engine.generate(WARMUP_PROMPT, sp, request_id="warmup")
                    async for _ in gen:
                        pass
                    print("vLLM warmup done")
                except Exception as e:
                    print("vLLM warmup failed (continue):", e)

            await _warmup_vllm()
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
                **tokenizer(WARMUP_PROMPT, return_tensors="pt").to(model.device),
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=TOP_K,
            )
            print("Transformers warmup done")
        except Exception as e:
            print("Warmup failed (can ignore):", e)

        app.state.tokenizer = tokenizer
        app.state.model = model
        print("Transformers model initialized successfully!")

    app.state.ready = True

    yield
    print("Shutting down...")


app = FastAPI(title="LLM Service", lifespan=lifespan)


@app.get("/")
def health_check():
    if not getattr(app.state, "ready", False):
        return {"status": "warming"}
    if BATCH_MODE:
        return {"status": "batch"}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
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
            temperature=0.0,
            top_p=1.0,
            top_k=TOP_K,
            max_tokens=MAX_NEW_TOKENS,
            frequency_penalty=0.0,
        )

        async def run_one(text_prompt: str) -> str:
            # vLLM 的 request_id 需要唯一；batch 并发下用 uuid 避免竞争条件
            rid = uuid.uuid4().hex
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
            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=TOP_K,
            )
            out = mdl.generate(**inputs, **gen_kwargs)
            text = tok.decode(out[0], skip_special_tokens=True)
            if text.startswith(pt):
                text = text[len(pt):].strip()
            outputs.append(strip_think(text))
        return PredictionResponse(response=outputs)

    prompt_text = prompt_texts[0]
    inputs = tok(prompt_text, return_tensors="pt").to(mdl.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=TOP_K,
    )

    out = mdl.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0], skip_special_tokens=True)

    if text.startswith(prompt_text):
        text = text[len(prompt_text):].strip()

    return PredictionResponse(response=strip_think(text))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
