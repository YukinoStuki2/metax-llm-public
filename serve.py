import os
import re
import shutil
import json
import logging
from contextlib import asynccontextmanager
import asyncio
import uuid
import inspect
from typing import Any, List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 避免 transformers 在启动时引入 torchvision（在部分环境会产生大量 warning，甚至可能触发版本不匹配错误）。
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# ======================
# 配置区：容器优先
# ======================
# Dockerfile / run_model.sh 下载到 ./model/$MODEL_ID；默认使用同一路径
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join("./model", os.environ.get("MODEL_ID", "YukinoStuki/Qwen3-4B-Plus-LLM")),
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

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
# 对“需要代码/实现”的题，允许更长输出，避免截断导致 RougeL 偏低。
# 注意：默认不把全局 MAX_NEW_TOKENS 拉大，以免短答题浪费 token、拖慢吞吐。
MAX_NEW_TOKENS_CODE = int(os.environ.get("MAX_NEW_TOKENS_CODE", "192"))
WARMUP_PROMPT = "你好"

# Batch 模式：GET / 返回 {"status":"batch"} 后，评测机会一次性把所有问题推到 /predict
# 注意：评测/容器环境可能会在不同位置注入环境变量；因此这里不要在 import 时固化，
# 统一通过函数动态读取，避免出现“明明开了 batch 但 health 仍返回 ok”的情况。
def is_batch_mode() -> bool:
    return os.environ.get("BATCH_MODE", "0") == "1"

# batch 并发：并发提交给 vLLM 以触发引擎内 batching
BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "320"))

# TEMPERATURE=0 -> 确定性生成（更稳更快）
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TOP_K = int(os.environ.get("TOP_K", "1"))


def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def is_code_question(user_prompt: str) -> bool:
    """启发式判断：题目是否更可能需要代码/实现类输出。

    plus 题里很多是 Triton/TileLang/CUDA/Kernel 相关，参考答案常给伪代码或代码片段。
    对这种题把 max_new_tokens 适当提高能显著减少截断。
    """

    p = _normalize_text(user_prompt)
    if not p:
        return False

    # 用户可通过环境变量追加关键词
    extra = os.environ.get("CODE_QUESTION_KEYWORDS", "")
    extra_words = [w.strip().lower() for w in extra.split(",") if w.strip()]

    keywords = [
        # 中文关键词
        "代码",
        "伪代码",
        "实现",
        "kernel",
        "算子",
        "cuda",
        "triton",
        "tilelang",
        "cudagraph",
        "共享内存",
        "shared memory",
        "线程",
        "block",
        "grid",
        "索引",
        "矩阵乘法",
        "卷积",
        "depthwise",
        "dilated",
        "空洞卷积",
        # 常见代码特征词
        "@triton",
        "tl.",
        "__global__",
        "#include",
        "import ",
        "def ",
        "for ",
        "while ",
        "return ",
    ]
    keywords.extend(extra_words)

    return any(k in p for k in keywords)


def pick_max_new_tokens(user_prompt: str) -> int:
    return MAX_NEW_TOKENS_CODE if is_code_question(user_prompt) else MAX_NEW_TOKENS


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

# vLLM 开关：auto/true/false
USE_VLLM = os.environ.get("USE_VLLM", "true").lower()

# 若设置为 1：vLLM 初始化失败时直接退出（不回退到 transformers）。
# 默认 0：vLLM 失败时回退，保证服务可启动。
FORCE_VLLM = os.environ.get("FORCE_VLLM", "0") == "1"

# ⚠️ 评测 run 阶段断网：不要每次 predict 进行联网探测（会拖慢）
DEBUG_NET = os.environ.get("DEBUG_NET", "0") == "1"

# vLLM 在部分环境可用 ModelScope（保留）
os.environ["VLLM_USE_MODELSCOPE"] = "True"

# 跨平台安全处理：
# - 在 MetaX 云（如 C500）上，可能存在 vllm_metax 平台插件且需要正常工作。
# - 在非 MetaX 机器（如 WSL + RTX4090）上，如果误装 vllm_metax，可能被自动激活，
#   反而导致 CUDA 执行异常。
# 若用户显式设置了 VLLM_PLUGINS，则尊重用户设置。
_HAS_MX_DEVICE = any(os.path.exists(p) for p in ("/dev/mxcd", "/dev/mxc0", "/dev/mxc"))
if "VLLM_PLUGINS" not in os.environ and not _HAS_MX_DEVICE:
    # 空字符串 => 不加载任何插件（vLLM 内置的 CUDA 探测仍可用）。
    os.environ["VLLM_PLUGINS"] = ""

# MetaX 默认：除非用户显式覆盖，否则保持启用 V1 引擎。
# 一些 MetaX 平台插件构建仅支持（或强依赖）V1 引擎路径；设置 VLLM_USE_V1=0
# 可能导致启动阶段直接报错。
if _HAS_MX_DEVICE and "VLLM_USE_V1" not in os.environ:
    os.environ["VLLM_USE_V1"] = "1"

# MetaX 上优先使用更保守的默认值（除非用户覆盖）。
if _HAS_MX_DEVICE and "GPU_MEMORY_UTILIZATION" not in os.environ:
    os.environ["GPU_MEMORY_UTILIZATION"] = "0.60"


def _unset_env_if_blank(key: str) -> None:
    """当环境变量存在但值为空时，移除它。

    一些云运行时会误把 CUDA_VISIBLE_DEVICES 设置为空字符串，导致 vLLM/torch
    报错（例如："Device string must not be empty"）。
    """
    if key in os.environ and os.environ.get(key, "").strip() == "":
        del os.environ[key]


# 防御性处理：避免空设备字符串导致 vLLM 异常。
_unset_env_if_blank("CUDA_VISIBLE_DEVICES")
_unset_env_if_blank("NVIDIA_VISIBLE_DEVICES")
_unset_env_if_blank("VLLM_DEVICE")


def _has_c_compiler() -> bool:
    # Triton（vLLM 依赖）可能在运行时编译少量 C/CUDA 辅助模块。
    # 若系统无编译器，vLLM 可能会在初始化阶段报错："Failed to find C compiler"。
    return any(shutil.which(x) for x in ("cc", "gcc", "clang"))

def strip_think(text: str) -> str:
    """去掉可能的 <think>...</think>，避免输出过长。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _maybe_parse_estimated_max_len(err: Exception) -> Optional[int]:
    """从 vLLM 的 KV-cache 相关报错中解析推荐的 max model length。

    注意：vLLM 有时会在父进程抛出较泛化的 RuntimeError，而子进程日志里才有更详细的
    ValueError；因此这里会同时扫描异常链（cause/context）。
    """

    def _iter_exc_chain(e: BaseException):
        seen: set[int] = set()
        cur: Optional[BaseException] = e
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            yield cur
            # 优先取显式 cause，否则退回 context。
            cur = cur.__cause__ or cur.__context__

    for ex in _iter_exc_chain(err):
        msg = str(ex)
        m = re.search(r"estimated maximum model length is\s+(\d+)", msg)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _maybe_fix_compilation_config(engine_kwargs: dict, err: Exception) -> bool:
    """跨 vLLM 版本兼容地修复 compilation_config 的类型。

    不同 vLLM 构建对 compilation_config 的期望可能是：
    - dict / CompilationConfig 数据类
    - 或者可被解析为 CompilationConfig 的 JSON 字符串

    这里根据常见的 Pydantic 报错模式，在 dict <-> JSON 字符串之间做一次转换。
    若 engine_kwargs 被修改则返回 True。
    """

    if "compilation_config" not in engine_kwargs:
        return False

    msg = str(err)
    if (
        "compilation_config" not in msg
        and "CompilationConfig" not in msg
        and "Invalid JSON" not in msg
        and "json_invalid" not in msg
    ):
        return False

    val = engine_kwargs.get("compilation_config")

    # 情况 A：期望 dict/数据类，但实际拿到的是 JSON 字符串。
    if isinstance(val, str) and (
        "Input should be a dictionary" in msg
        or "dataclass_type" in msg
    ):
        try:
            engine_kwargs["compilation_config"] = json.loads(val)
            return True
        except Exception:
            return False

    # 情况 B：期望 JSON 字符串，但实际拿到的是 dict。
    if isinstance(val, dict) and (
        "Invalid JSON" in msg
        or "json_invalid" in msg
        or "Input should be a valid string" in msg
    ):
        engine_kwargs["compilation_config"] = json.dumps(val)
        return True

    # 情况 C：拿到的是 Python dict 字符串（如 "{'level': 0}"）；转换为合法 JSON 字符串。
    if isinstance(val, str) and ("Invalid JSON" in msg or "json_invalid" in msg):
        try:
            parsed = json.loads(val)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            engine_kwargs["compilation_config"] = json.dumps(parsed)
            return True

        try:
            import ast

            parsed2 = ast.literal_eval(val)
            if isinstance(parsed2, dict):
                engine_kwargs["compilation_config"] = json.dumps(parsed2)
                return True
        except Exception:
            return False

    return False


class PredictionRequest(BaseModel):
    # 单条：str；batch：list[str]
    prompt: Union[str, List[str]]


class PredictionResponse(BaseModel):
    # 单条：str；batch：list[str]
    response: Union[str, List[str]]


def _coerce_log_level(level_name: str, default: int = logging.WARNING) -> int:
    name = (level_name or "").strip().upper()
    return int(getattr(logging, name, default))


def _set_logger_level_prefix(prefix: str, level: int) -> None:
    """为指定 logger 前缀设置日志等级（例如 'vllm'）。

    vLLM 在 batch 模式下可能非常“话痨”（例如每个请求都会打印 INFO：'Added request ...'）。
    这些日志对评测无收益，反而增加开销，因此默认把 vLLM 相关日志降到 WARNING。
    """

    try:
        logging.getLogger(prefix).setLevel(level)
        # 同步更新可能已创建的子 logger。
        for name in list(getattr(logging.root.manager, "loggerDict", {}).keys()):
            if name == prefix or name.startswith(prefix + "."):
                logging.getLogger(name).setLevel(level)
    except Exception:
        pass


# ======================
# 可选依赖探测
# ======================
_vllm_ok = False
_transformers_ok = False
_vllm_v1_ok = False

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    _vllm_ok = True

    _vllm_level = _coerce_log_level(os.environ.get("VLLM_LOG_LEVEL", "WARNING"))
    _set_logger_level_prefix("vllm", _vllm_level)
except Exception:
    _vllm_ok = False

try:
    # vLLM V1 引擎（开发预览/新版）：AsyncLLM
    # 注意：不同构建可能不包含 v1 包或接口有差异，因此必须可选导入。
    from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore

    _vllm_v1_ok = True
except Exception:
    _vllm_v1_ok = False


def _get_vllm_engine_mode() -> str:
    """选择 vLLM 推理引擎实现。

    - v0: 现有 AsyncLLMEngine（默认，最稳）
    - v1: 新版 AsyncLLM（可能更快/更省 CPU 调度开销，但兼容性风险更高）

    说明：我们不复用 VLLM_USE_V1 环境变量作为选择条件，因为该变量可能被平台/插件用作
    vLLM 内部行为开关；为了不改变既有语义，这里单独用 SERVE_VLLM_ENGINE。
    """

    return (os.environ.get("SERVE_VLLM_ENGINE") or "v0").strip().lower()


def _use_vllm_offline_llm_in_batch_mode() -> bool:
    """是否在 batch 模式下改用 vLLM 的离线 LLM 批推理入口。

    背景：当前 /predict(batch) 通过 asyncio 并发 N 次 `engine.generate(...)` 来触发 vLLM
    内部 batching，但 Python 层会创建大量 task + 消费大量 async generator，存在可观调度开销。

    vLLM 的 `LLM.generate(list_prompts)` 是面向离线批推理设计的接口，单次调用即可把所有
    prompt 提交给引擎，通常能更充分地吃满引擎的连续批处理能力。

    说明：该模式仅在 BATCH_MODE=1 时默认启用；也可通过环境变量显式开关。
    """

    if not is_batch_mode():
        return False
    return os.environ.get("VLLM_BATCH_USE_LLM", "1") == "1"

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _transformers_ok = True
except Exception:
    _transformers_ok = False


def should_use_vllm() -> bool:
    if not _vllm_ok:
        return False

    # 若已知当前环境缺少工具链会导致 vLLM 失败，则优先 transformers；
    # 除非用户显式强制使用 vLLM。
    if not FORCE_VLLM and not _has_c_compiler():
        return False

    # 只有在 CUDA 实际可用时才倾向使用 vLLM。
    cuda_ok = False
    try:
        import torch  # type: ignore

        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if USE_VLLM == "true":
        # 强制尝试 vLLM；初始化失败则回退到 transformers。
        return True
    if USE_VLLM == "false":
        return False
    return cuda_ok


def build_prompt(user_prompt: str) -> str:
    # 兼容：旧逻辑（不会走 chat template 的情况下）
    return f"{SYSTEM_PROMPT}\n问题：{user_prompt}\n答案："


def format_as_chat(tokenizer: Any, user_prompt: str) -> str:
    """用 Qwen 系列的 chat template 构造最终 prompt（提升准确率/一致性）。"""
    p = (user_prompt or "").strip()

    # batch 模式下，1024+ 样本的 apply_chat_template(Jinja) CPU 开销可能很可观。
    # 对于 Qwen3/2.x 常见的「system + user + assistant 开头」结构，我们做一个等价的
    # 快路径：直接字符串拼接，显著减少 Python/Jinja 调度成本。
    # 若判断不成立则回退到 tokenizer.apply_chat_template，保证兼容性。
    def _supports_im_start_template(tok: Any) -> bool:
        try:
            tpl = getattr(tok, "chat_template", None)
            return isinstance(tpl, str) and ("<|im_start|>" in tpl) and ("assistant" in tpl)
        except Exception:
            return False

    fast_chat = _env_flag("FAST_CHAT_TEMPLATE", default=is_batch_mode())
    if fast_chat and tokenizer is not None and _supports_im_start_template(tokenizer):
        # 与仓库内 Qwen3 chat_template.jinja 的常见路径等价（无 tools）。
        # system：'<|im_start|>system\n' + content + '<|im_end|>\n'
        # user  ：'<|im_start|>user\n' + content + '<|im_end|>\n'
        # gen   ：'<|im_start|>assistant\n'
        return (
            "<|im_start|>system\n"
            + SYSTEM_PROMPT
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + p
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )

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
    # 本地友好兜底：若用户未设置 MODEL_DIR，且默认 ./model/$MODEL_ID 不存在，
    # 但 ./merged 存在，则使用 ./merged。
    if "MODEL_DIR" not in os.environ:
        default_dir = os.path.abspath(MODEL_DIR)
        merged_dir = os.path.abspath("./merged")
        if (not os.path.isdir(default_dir)) and os.path.isdir(merged_dir):
            abs_model_dir = merged_dir
        else:
            abs_model_dir = default_dir
    else:
        abs_model_dir = os.path.abspath(MODEL_DIR)
    print("MODEL_DIR =", abs_model_dir)
    print("BATCH_MODE =", os.environ.get("BATCH_MODE", "0"))

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
            # 云环境诊断信息（便于定位驱动/可见卡等问题）
            try:
                import torch as _torch  # type: ignore

                print("torch.cuda.is_available() =", _torch.cuda.is_available())
                if _torch.cuda.is_available():
                    print("torch.cuda.device_count() =", _torch.cuda.device_count())
            except Exception as e:
                print("torch cuda probe failed:", e)

            print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
            print("NVIDIA_VISIBLE_DEVICES =", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
            print("VLLM_DEVICE =", os.environ.get("VLLM_DEVICE"))
            print("VLLM_PLUGINS =", os.environ.get("VLLM_PLUGINS"))
            print("HAS_MX_DEVICE =", _HAS_MX_DEVICE)

            # 注意：尽量只使用通用/稳定参数，避免不同 vLLM 版本不兼容
            engine_kwargs = dict(
                model=abs_model_dir,
                tensor_parallel_size=1,
                gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")),
                trust_remote_code=True,
                dtype=os.environ.get("DTYPE", "float16"),
                disable_log_stats=True,
            )

            # 可选：覆盖最大序列长度（在小显存 GPU 上有助于避免 KV-cache OOM）。
            max_model_len_env = os.environ.get("MAX_MODEL_LEN")

            # 部分 vLLM 版本/平台插件不接受 `device` 参数。
            try:
                sig = inspect.signature(AsyncEngineArgs.__init__)

                # 可选：量化/加载格式透传（仅在当前 vLLM 构建支持时设置）
                # 例如：VLLM_QUANTIZATION=awq, VLLM_LOAD_FORMAT=awq
                vllm_quant = (os.environ.get("VLLM_QUANTIZATION") or "").strip()
                if vllm_quant and "quantization" in sig.parameters:
                    engine_kwargs["quantization"] = vllm_quant
                    print("Using VLLM_QUANTIZATION =", vllm_quant)

                vllm_load_format = (os.environ.get("VLLM_LOAD_FORMAT") or "").strip()
                if vllm_load_format and "load_format" in sig.parameters:
                    engine_kwargs["load_format"] = vllm_load_format
                    print("Using VLLM_LOAD_FORMAT =", vllm_load_format)

                # 吞吐相关开关（仅在当前 vLLM 构建支持时才设置）
                # 前缀缓存通常是安全的，且能显著加速“共享前缀”的大 batch（如 system prompt）。
                if "enable_prefix_caching" in sig.parameters:
                    engine_kwargs["enable_prefix_caching"] = _env_flag("ENABLE_PREFIX_CACHING", True)
                if "disable_log_requests" in sig.parameters:
                    engine_kwargs["disable_log_requests"] = True

                # 可选：容量调参（默认不设置；允许通过环境变量覆盖）
                max_num_seqs_env = os.environ.get("VLLM_MAX_NUM_SEQS")
                if max_num_seqs_env and "max_num_seqs" in sig.parameters:
                    try:
                        engine_kwargs["max_num_seqs"] = int(max_num_seqs_env)
                        print("Using VLLM_MAX_NUM_SEQS =", engine_kwargs["max_num_seqs"])
                    except Exception:
                        pass

                max_batched_tokens_env = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS")
                if max_batched_tokens_env and "max_num_batched_tokens" in sig.parameters:
                    try:
                        engine_kwargs["max_num_batched_tokens"] = int(max_batched_tokens_env)
                        print("Using VLLM_MAX_NUM_BATCHED_TOKENS =", engine_kwargs["max_num_batched_tokens"])
                    except Exception:
                        pass

                # MetaX 驱动栈可能对新版 vLLM 的高级特性较敏感（V1 引擎、编译、chunked prefill）。
                # 这里采用“先性能、失败再回退”的策略：
                # - 默认不强制 eager（这样 cudagraph 仍有机会生效，吞吐更高）
                # - 默认限制 max_model_len，避免直接按模型 config 的超长上下文（如 262144）分配 KV cache，导致并发极低
                # - 若初始化失败，再自动回退到更保守设置
                if _HAS_MX_DEVICE:
                    vllm_enforce_eager = _env_flag("VLLM_ENFORCE_EAGER", False)
                    if "enforce_eager" in sig.parameters and vllm_enforce_eager:
                        engine_kwargs["enforce_eager"] = True

                    # 用户可显式设 MAX_MODEL_LEN；否则 MetaX 默认用一个更保守的长度（评测 prompt 通常不需要超长上下文）。
                    if max_model_len_env is None or str(max_model_len_env).strip() == "":
                        default_len = os.environ.get("DEFAULT_MAX_MODEL_LEN", "38400")
                        if "max_model_len" in sig.parameters:
                            try:
                                engine_kwargs["max_model_len"] = int(default_len)
                                print("Using DEFAULT_MAX_MODEL_LEN =", engine_kwargs["max_model_len"])
                            except Exception:
                                pass

                    if "enable_chunked_prefill" in sig.parameters:
                        # MetaX 上默认禁用 chunked prefill（保守），可通过环境变量显式开启。
                        engine_kwargs["enable_chunked_prefill"] = _env_flag(
                            "ENABLE_CHUNKED_PREFILL",
                            False,
                        )

                    if "compilation_config" in sig.parameters:
                        # 默认不要设置 compilation_config。
                        # 不同 vLLM 构建/插件对该参数接受的类型（dict vs JSON 字符串）可能不一致，
                        # 错误的类型转换会导致启动失败。若用户确实需要覆盖，请通过
                        # VLLM_COMPILATION_CONFIG 提供 JSON 字符串。
                        cfg = os.environ.get("VLLM_COMPILATION_CONFIG")
                        if cfg:
                            engine_kwargs["compilation_config"] = cfg

                if max_model_len_env is not None and "max_model_len" in sig.parameters:
                    try:
                        engine_kwargs["max_model_len"] = int(max_model_len_env)
                        print("Using MAX_MODEL_LEN =", engine_kwargs["max_model_len"])
                    except Exception:
                        pass

                if "device" in sig.parameters:
                    vllm_device = (os.environ.get("VLLM_DEVICE") or "cuda").strip() or "cuda"
                    engine_kwargs["device"] = vllm_device
                    print("AsyncEngineArgs supports device =", vllm_device)
            except Exception:
                pass

            use_offline_llm = _use_vllm_offline_llm_in_batch_mode()
            engine_mode = _get_vllm_engine_mode()

            def _build_engine() -> AsyncLLMEngine:
                engine_args = AsyncEngineArgs(**engine_kwargs)
                return AsyncLLMEngine.from_engine_args(engine_args)

            def _build_engine_v1():
                # V1 引擎：AsyncLLM
                # generate 签名为 generate(request_id=..., prompt=..., sampling_params=...)
                engine_args = AsyncEngineArgs(**engine_kwargs)
                return AsyncLLM.from_engine_args(engine_args)  # type: ignore

            def _build_llm():
                # 注意：LLM 是同步接口，适合离线/大 batch；我们会在 /predict 中用 to_thread 包装。
                # 这里尽量复用与 AsyncEngineArgs 一致的参数集（LLM 的 **kwargs 会转发给 EngineArgs）。
                from vllm import LLM  # 延迟导入，避免在不需要时引入额外依赖链

                llm_kwargs = dict(engine_kwargs)
                return LLM(**llm_kwargs)

            try:
                if use_offline_llm:
                    print("BATCH_MODE=1: using vLLM LLM.generate(list_prompts) path")
                    app.state.llm = _build_llm()
                    app.state.llm_lock = asyncio.Lock()
                    app.state.engine = None
                    app.state.engine_kind = "llm"
                else:
                    if engine_mode == "v1" and _vllm_v1_ok:
                        print("Using vLLM V1 engine (AsyncLLM)")
                        app.state.engine = _build_engine_v1()
                        app.state.engine_kind = "async_v1"
                    else:
                        if engine_mode == "v1" and not _vllm_v1_ok:
                            print("SERVE_VLLM_ENGINE=v1 requested but v1 engine not available; fallback to v0")
                        app.state.engine = _build_engine()
                        app.state.engine_kind = "async_v0"
                    app.state.llm = None
                    app.state.llm_lock = None
            except Exception as e:
                # 若 LLM 路线失败，自动回退到 AsyncLLMEngine（尽量不影响可用性）。
                if use_offline_llm:
                    print("vLLM LLM init failed, fallback to AsyncLLMEngine. Error:", e)
                    try:
                        app.state.engine = _build_engine()
                        app.state.llm = None
                        app.state.llm_lock = None
                        app.state.engine_kind = "async_v0"
                        engine_built = True
                        last_err = e
                    except Exception as e_engine:
                        last_err = e_engine
                        engine_built = False
                else:
                    # 若 V1 引擎失败，自动回退到 V0 引擎
                    if engine_mode == "v1":
                        print("vLLM V1 engine init failed, fallback to v0. Error:", e)
                        try:
                            app.state.engine = _build_engine()
                            app.state.engine_kind = "async_v0"
                            app.state.llm = None
                            app.state.llm_lock = None
                            engine_built = True
                            last_err = e
                        except Exception as e_engine:
                            last_err = e_engine
                            engine_built = False
                    else:
                        last_err = e
                        engine_built = False

                # 仅 AsyncLLMEngine 路线需要这组回退。
                if not use_offline_llm:
                    # 1) 若 compilation_config 来自环境变量覆盖，则跨版本尝试自动修正一次类型。
                    if _HAS_MX_DEVICE and _maybe_fix_compilation_config(engine_kwargs, last_err):
                        try:
                            app.state.engine = _build_engine()
                            engine_built = True
                        except Exception as e2:
                            last_err = e2

                # 2) MetaX：如果非 eager 初始化失败，自动回退为 eager（更保守但通常更稳）。
                # 说明：你日志里 enforce_eager=True 会禁用 cudagraph，吞吐可能下降；
                # 因此这里优先让性能模式成功，失败再回退。
                if (not use_offline_llm) and (not engine_built):
                    try:
                        sig = inspect.signature(AsyncEngineArgs.__init__)
                        can_set_eager = ("enforce_eager" in sig.parameters)
                    except Exception:
                        can_set_eager = False

                    if _HAS_MX_DEVICE and can_set_eager and ("enforce_eager" not in engine_kwargs):
                        print("Retry vLLM init with enforce_eager = True")
                        engine_kwargs["enforce_eager"] = True
                        # 回退时也保持 chunked prefill 默认关闭（除非用户显式开启）。
                        if "enable_chunked_prefill" in engine_kwargs:
                            engine_kwargs["enable_chunked_prefill"] = _env_flag(
                                "ENABLE_CHUNKED_PREFILL",
                                False,
                            )
                        try:
                            app.state.engine = _build_engine()
                            engine_built = True
                        except Exception as e3:
                            last_err = e3

                # 3) 若 KV cache 不足以支撑模型配置的最大序列长度，且 vLLM 支持设置 max_model_len，
                # 则降低长度重试一次。
                if (not use_offline_llm) and (not engine_built):
                    suggested_len = _maybe_parse_estimated_max_len(last_err)
                    try:
                        sig = inspect.signature(AsyncEngineArgs.__init__)
                        can_set_len = ("max_model_len" in sig.parameters)
                    except Exception:
                        can_set_len = False

                    if can_set_len and ("max_model_len" not in engine_kwargs):
                        retry_len: Optional[int] = None
                        if suggested_len is not None:
                            retry_len = int(suggested_len)
                        else:
                            # 一些 vLLM 版本会把 KV-cache 详细错误包装成泛化的 EngineCore failure；
                            # 此时尝试一个保守默认值。
                            msg = str(last_err)
                            if "Engine core initialization failed" in msg or "KV cache" in msg:
                                try:
                                    retry_len = int(os.environ.get("SAFE_MAX_MODEL_LEN", "38400"))
                                except Exception:
                                    retry_len = 38400

                        if retry_len is not None:
                            print("Retry vLLM init with max_model_len =", retry_len)
                            engine_kwargs["max_model_len"] = int(retry_len)
                            try:
                                app.state.engine = _build_engine()
                                engine_built = True
                            except Exception as e4:
                                last_err = e4

                if not engine_built:
                    raise last_err
            app.state.backend = "vllm"
            print("vLLM backend initialized successfully!")

            async def _warmup_vllm():
                try:
                    sp = SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=TOP_K,
                        max_tokens=8,
                        frequency_penalty=0.0,
                    )
                    if getattr(app.state, "llm", None) is not None:
                        llm = app.state.llm

                        def _do_warmup():
                            _ = llm.generate([WARMUP_PROMPT], sampling_params=sp, use_tqdm=False)

                        # 同步接口：放到线程池里预热
                        await asyncio.to_thread(_do_warmup)
                    else:
                        engine_kind = getattr(app.state, "engine_kind", "async_v0")
                        if engine_kind == "async_v1":
                            gen = app.state.engine.generate(
                                request_id="warmup",
                                prompt=WARMUP_PROMPT,
                                sampling_params=sp,
                            )
                        else:
                            gen = app.state.engine.generate(WARMUP_PROMPT, sp, request_id="warmup")
                        async for _ in gen:
                            pass
                    print("vLLM warmup done")
                except Exception as e:
                    print("vLLM warmup failed (continue):", e)

            await _warmup_vllm()
        except Exception as e:
            print("vLLM init failed, fallback to transformers. Error:", e)
            if FORCE_VLLM:
                raise
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
    if is_batch_mode():
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
    per_max_tokens = [pick_max_new_tokens(p) for p in prompts]
    backend = getattr(app.state, "backend", "transformers")

    if backend == "vllm":
        # 若初始化阶段选择了离线 LLM 路线（主要用于 BATCH_MODE=1 的极限吞吐），
        # 则单次调用 LLM.generate(list_prompts) 完成整个 batch。
        llm = getattr(app.state, "llm", None)
        if llm is not None:
            sampling_params_cache: dict[int, SamplingParams] = {}

            def get_sampling_params(max_tokens: int) -> SamplingParams:
                mt = int(max_tokens)
                sp = sampling_params_cache.get(mt)
                if sp is None:
                    sp = SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=TOP_K,
                        max_tokens=mt,
                        frequency_penalty=0.0,
                    )
                    sampling_params_cache[mt] = sp
                return sp

            params_list = [get_sampling_params(mt) for mt in per_max_tokens]
            llm_lock = getattr(app.state, "llm_lock", None)

            async def _run_llm_batch() -> List[str]:
                def _do_generate() -> List[str]:
                    outs = llm.generate(prompt_texts, sampling_params=params_list, use_tqdm=False)
                    results: List[str] = []
                    for o in outs:
                        if not getattr(o, "outputs", None):
                            results.append("")
                            continue
                        results.append(strip_think(o.outputs[0].text))
                    return results

                if llm_lock is None:
                    return await asyncio.to_thread(_do_generate)
                async with llm_lock:
                    return await asyncio.to_thread(_do_generate)

            outputs = await _run_llm_batch()
            if len(outputs) == 1 and isinstance(req.prompt, str):
                return PredictionResponse(response=outputs[0])
            return PredictionResponse(response=outputs)

        # 默认：AsyncLLMEngine 路线（异步流式，靠并发触发引擎 batching）
        engine = app.state.engine
        engine_kind = getattr(app.state, "engine_kind", "async_v0")

        sampling_params_cache: dict[int, SamplingParams] = {}

        def get_sampling_params(max_tokens: int) -> SamplingParams:
            mt = int(max_tokens)
            sp = sampling_params_cache.get(mt)
            if sp is None:
                sp = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=TOP_K,
                    max_tokens=mt,
                    frequency_penalty=0.0,
                )
                sampling_params_cache[mt] = sp
            return sp

        async def run_one(text_prompt: str, max_tokens: int) -> str:
            # vLLM 的 request_id 需要唯一；batch 并发下用 uuid 避免竞争条件
            rid = uuid.uuid4().hex
            if engine_kind == "async_v1":
                results = engine.generate(
                    request_id=rid,
                    prompt=text_prompt,
                    sampling_params=get_sampling_params(max_tokens),
                )
            else:
                results = engine.generate(text_prompt, get_sampling_params(max_tokens), rid)
            final_output = None
            async for request_output in results:
                final_output = request_output
            if final_output is None or not final_output.outputs:
                return ""
            return strip_think(final_output.outputs[0].text)

        # 单条：直接跑；batch：限制并发以触发 vLLM 内部 batching
        if len(prompt_texts) == 1:
            return PredictionResponse(response=await run_one(prompt_texts[0], per_max_tokens[0]))

        sem = asyncio.Semaphore(max(1, BATCH_CONCURRENCY))

        async def guarded(tp: str, mt: int) -> str:
            async with sem:
                return await run_one(tp, mt)

        outputs = await asyncio.gather(*[guarded(tp, mt) for tp, mt in zip(prompt_texts, per_max_tokens)])
        return PredictionResponse(response=outputs)

    # transformers 路线
    tok = app.state.tokenizer
    mdl = app.state.model

    if len(prompt_texts) != 1:
        # transformers 路线不建议 batch（慢且占显存），但为了兼容协议仍给出串行结果
        outputs: List[str] = []
        for pt, mt in zip(prompt_texts, per_max_tokens):
            inputs = tok(pt, return_tensors="pt").to(mdl.device)
            gen_kwargs = dict(
                max_new_tokens=int(mt),
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
    mt = per_max_tokens[0]

    gen_kwargs = dict(
        max_new_tokens=int(mt),
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
