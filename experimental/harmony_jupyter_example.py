# -*- coding: utf-8 -*-
"""
Refactored GPT-OSS-20B Harmony runner with tooling + KV cache reuse,
compatible with recent `transformers` (>=4.57).

Key changes vs the original script:

1. Use the official `DynamicCache` API instead of manually slicing
   and poking into `past_key_values`. This makes the code robust to
   internal changes in Transformers (incl. GPT-OSS sliding attention).
2. Implement a simpler, prefix-based KV reuse strategy:
   - Reuse the cache only when the new prompt token stream is a
     *strict prefix extension* of the previous one.
   - Otherwise, rebuild the cache from scratch.
   Note: we build prompts from the *actual token history* (including any
   non-canonical Harmony headers the model emitted) to avoid parse→render
   normalization breaking token-level KV reuse.
3. Follow Harmony’s recommendation to *not* feed stop tokens
   (`<|return|>`, `<|call|>`, etc.) into the parser, and treat
   `<|return|>` as a decode-time stop only.
4. Improve modularity: split config, runtime setup, KV cache logic,
   generation, and tool handling into small, testable pieces.
"""

# Suggested dependencies (GPU with SM >= 7.5 for MXFP4 kernels, e.g. RTX A5000):
#   pip install -U "torch>=2.8.0" "triton>=3.4.0" "transformers>=4.57.1" \
#       kernels accelerate openai-harmony gpt-oss jupyter_client ipykernel pyzmq

import asyncio
import inspect
import json
import sys
import threading
import gc
import time
import os
import codecs
import atexit
import logging
from dataclasses import dataclass
from collections.abc import Iterable, Callable
from datetime import date
from typing import Any, List, Optional

import torch
from jupyter_client import KernelManager
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    Author,
    SystemContent,
    load_harmony_encoding,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)
from gpt_oss.tools.simple_browser import SimpleBrowserTool, ExaBackend

# Prefer the repo-default Docker Python tool if available; fall back to Jupyter
try:
    from gpt_oss.tools.python_docker.docker_tool import PythonTool as DockerPythonTool  # type: ignore
except Exception:  # noqa: BLE001
    DockerPythonTool = None  # type: ignore


# =========================
# 0) 全局状态：模型 / 分词器 / 编码器 / 工具
# =========================

model_id = "openai/gpt-oss-20b"

tokenizer = None          # type: ignore[assignment]
model = None              # type: ignore[assignment]
encoding = None           # type: ignore[assignment]
browser_tool = None       # type: ignore[assignment]
python_tool = None        # type: ignore[assignment]
_decode_token_bytes: Callable[[int], bytes] | None = None
_assistant_start_token_ids: List[int] | None = None


# =========================
# 1) 运行时配置
# =========================

@dataclass
class RuntimeConfig:
    prefill_chunk: int = 128          # prefill 时的最大 chunk 大小（token 数）
    decode_release_every: int = 256   # 每多少 decode 步主动清一次 CUDA cache（0 = 不清）
    aggressive_empty_cache: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    seed: Optional[int] = None
    debug: bool = False


cfg: RuntimeConfig | None = None

logger = logging.getLogger("harmony_gpt_oss_runner")
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stderr)
    _fmt = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _load_config() -> RuntimeConfig:
    def _get_bool(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return val.strip().lower() in {"1", "true", "yes", "on"}

    def _get_int(name: str, default: int) -> int:
        v = os.getenv(name)
        try:
            return int(v) if v is not None else default
        except Exception:
            return default

    def _get_float(name: str, default: float) -> float:
        v = os.getenv(name)
        try:
            return float(v) if v is not None else default
        except Exception:
            return default

    seed_env = os.getenv("SEED")
    seed_val: Optional[int] = None
    if seed_env is not None:
        try:
            seed_val = int(seed_env)
        except Exception:
            seed_val = None

    return RuntimeConfig(
        prefill_chunk=_get_int("PREFILL_CHUNK", 128),
        aggressive_empty_cache=_get_bool("AGGRESSIVE_EMPTY_CACHE", True),
        decode_release_every=_get_int("DECODE_RELEASE_EVERY", 256),
        temperature=_get_float("TEMPERATURE", 1.0),
        top_p=_get_float("TOP_P", 1.0),
        seed=seed_val,
        debug=_get_bool("DEBUG", False),
    )


# =========================
# 2) 小工具
# =========================

def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            torch.cuda.empty_cache()


def _build_token_byte_decoder() -> Callable[[int], bytes]:
    """
    Construct a fast token-id → raw-bytes decoder using openai-harmony's
    internal tiktoken-rs binding.

    This is the same trick as in the original script, but wrapped in a
    tiny helper.
    """
    assert encoding is not None, "Encoding not initialized. Call setup_runtime() first."
    inner = getattr(encoding, "_inner", None)
    if inner is None or not hasattr(inner, "decode_bytes"):
        raise AttributeError("encoding._inner.decode_bytes not available")

    decode_bytes = inner.decode_bytes
    return lambda token_id: bytes(decode_bytes([token_id]))


def _softmax_sample_top_p(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    import torch.nn.functional as F

    if temperature <= 0:
        temperature = 1e-6
    logits = (logits / temperature).to(torch.float32)
    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask = cdf > top_p
        if bool(mask[0].item()):
            mask[0] = False
        sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_id = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[next_id].item())
    else:
        return int(torch.multinomial(probs, num_samples=1).item())


def _content_to_code_str(msg: Message) -> str:
    """从 Harmony Message 抽取 python 源码字符串，兼容 JSON 包装 / code 片段等."""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and isinstance(obj.get("code"), str):
                return obj["code"]
        except Exception:
            return c

    try:
        if isinstance(c, Iterable) and not isinstance(c, (bytes, bytearray, str)):
            parts: List[str] = []
            for p in c:
                if hasattr(p, "code") and isinstance(p.code, str):
                    parts.append(p.code)
                elif hasattr(p, "text") and isinstance(p.text, str):
                    parts.append(p.text)
            if parts:
                return "\n".join(parts)
    except Exception:
        pass

    t = getattr(c, "text", None)
    return t if isinstance(t, str) else str(c)


def _content_to_text(content: Any) -> str:
    """把 Harmony 内容统一成纯文本（处理 TextContent 列表等）."""
    if isinstance(content, str):
        return content
    try:
        if isinstance(content, Iterable) and not isinstance(content, (bytes, bytearray, str)):
            parts: List[str] = []
            for p in content:
                t = getattr(p, "text", None)
                parts.append(t if isinstance(t, str) else str(p))
            return "\n".join(parts)
    except Exception:
        pass
    t = getattr(content, "text", None)
    if isinstance(t, str):
        return t
    return str(content)


# =========================
# 3) Python 工具（Jupyter 内核）
# =========================


class JupyterPythonTool:
    """
    使用本机 Jupyter kernel 作为 GPT-OSS 的 python 工具后端（有状态）。
    - 工具名固定为 'python'
    - 单实例维持一个 kernel，会话内多次调用共享变量/文件
    """

    name = "python"

    def __init__(self, kernel_name: str = "python3", timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self.km = KernelManager(kernel_name=kernel_name)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        # 确保 kernel 就绪（容错处理）
        try:
            self.kc.wait_for_ready(timeout=30)
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            self.kc.stop_channels()
        finally:
            try:
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass

    def _run_code(self, code: str) -> dict:
        msg_id = self.kc.execute(code, allow_stdin=False, stop_on_error=True)
        t0 = time.time()
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        displays: List[dict] = []
        last_text_result: Optional[str] = None

        while True:
            if time.time() - t0 > self.timeout_s:
                stderr_parts.append(f"\n[Timeout] execution exceeded {self.timeout_s}s")
                try:
                    self.km.interrupt_kernel()
                except Exception:
                    pass
                break

            try:
                msg = self.kc.get_iopub_msg(timeout=0.2)
            except Exception:
                continue

            mtype = msg["header"]["msg_type"]
            content = msg.get("content", {})

            if mtype == "stream":
                if content.get("name") == "stdout":
                    stdout_parts.append(content.get("text", ""))
                elif content.get("name") == "stderr":
                    stderr_parts.append(content.get("text", ""))
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "text/plain" in data:
                    last_text_result = data["text/plain"]
                if "image/png" in data:
                    displays.append({"mime": "image/png", "data": data["image/png"]})
                if "text/html" in data:
                    displays.append({"mime": "text/html", "data": data["text/html"]})
            elif mtype == "error":
                tb = "\n".join(content.get("traceback", []))
                stderr_parts.append(tb or f"{content.get('ename','')}: {content.get('evalue','')}")
            elif mtype == "status" and content.get("execution_state") == "idle":
                break

        return {
            "stdout": "".join(stdout_parts),
            "stderr": "".join(stderr_parts),
            "result": last_text_result,
            "displays": displays,
            "files": [],
        }

    def process(self, py_call_msg: Message) -> List[Message]:
        code = _content_to_code_str(py_call_msg)
        payload = self._run_code(code)
        tool_msg = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, "python"),
                json.dumps(payload),
            )
            .with_channel("commentary")
            .with_recipient("assistant")
            .with_content_type("json")
        )
        return [tool_msg]


def coerce_python_call_message(last_msg: Message) -> Message:
    """
    把“模型发起的工具调用”规范化为 PythonTool 可执行的格式。
    模型可能传回 JSON 如 {"code": "print(2+2)"}，
    而 PythonTool 期望是纯 Python 源码。
    """
    raw = getattr(last_msg, "content", "")
    if not isinstance(raw, str):
        raw = _content_to_text(raw)

    code = raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "code" in obj:
            code = obj["code"]
    except Exception:
        pass

    return (
        Message.from_role_and_content(Role.ASSISTANT, code)
        .with_channel("commentary")
        .with_recipient("python")
    )


# =========================
# 4) KV Cache 管理：基于 DynamicCache 的前缀缓存
# =========================


class KvCacheManager:
    """
    KV 缓存管理（带真实日志与统计版）

    约定：
    - self.prefix_ids：上一轮结束时，模型已经见过的完整 token 序列
      （包括 prefill + decode 的 token）。
    - self.total_len：与 prefix_ids 长度保持一致，表示当前 cache 中的
      逻辑序列长度（不关心 sliding-window 内部如何截断）。
    - 只有当“新 input_ids 是旧 prefix_ids 的严格前缀扩展”时，
      才会复用前缀对应的 KV；否则整体重建。

    注意 / 假设：
    - 上层对话是“只追加不回写”：不会修改历史消息内容，只会在末尾 append
      新的 user / assistant / tool 消息。
      若该假设不成立，新 input_ids 可能不再以旧 prefix 为前缀，此时本类会自动 reset，
      放弃 KV 复用，保证语义正确。

    额外说明：
    - 实测某些模型会输出“等价但非规范”的 Harmony message header（例如把 `to=python` 写进
      channel 字段），parser 会规范化该结构，导致 parse→render 的 token 流不再一致。
      因此本 runner 使用 kv_manager.history_ids 维护“真实 token 历史”，避免无谓的 cache 重建。
    """

    def __init__(self, model: AutoModelForCausalLM, prefill_chunk: int = 128):
        self.model = model
        self.prefill_chunk = max(1, int(prefill_chunk))

        self.cache: Optional[DynamicCache] = None
        self.prefix_ids: List[int] = []
        self.total_len: int = 0
        # 已确认“写入 prompt 的真实 token 流”（包含模型曾生成的非规范 header 形式）。
        # 用它来构造下一轮 prompt，避免 parse→render 规范化导致 token 不一致从而破坏 KV 复用。
        self.history_ids: List[int] = []

        # 简单的统计信息
        self.stats: dict[str, int] = {
            "prefill_calls": 0,
            "prefill_tokens_total": 0,
            "prefill_tokens_forwarded": 0,
            "prefill_tokens_reused": 0,
            "decode_tokens": 0,
        }

    def reset(self) -> None:
        self.cache = DynamicCache(config=self.model.config)
        self.prefix_ids = []
        self.total_len = 0

    @torch.inference_mode()
    def prefill_to(self, input_ids: List[int]) -> torch.Tensor:
        """
        确保 cache 覆盖完整的 input_ids 序列，并返回最后一个 token 的 logits。

        - 仅当“旧 prefix 完全是新 prompt 的前缀”时复用 KV；
        - 否则整段重建；
        - 同时记录详细日志与统计。
        """
        if not input_ids:
            raise ValueError("prefill_to expects non-empty input_ids")

        # --- 1) 记录旧状态，用于日志 ---
        old_prefix_len = len(self.prefix_ids)
        old_total_len = self.total_len

        # 新旧序列的最长公共前缀长度（仅用于日志，逻辑上不依赖它）
        lcp = 0
        for a, b in zip(self.prefix_ids, input_ids):
            if a != b:
                break
            lcp += 1

        reset_reason: str | None = None
        can_try_reuse = True

        # --- 2) 基本不变量检查：cache 未就绪 / 长度不一致 → 直接 reset ---
        # 额外：若 DynamicCache 内部记录的长度与 total_len 不一致，也直接 reset，
        # 防止未来有其他代码路径修改了 cache 却忘记同步 meta。
        real_lens: list[int] = []
        if self.cache is not None:
            # 对 hybrid 结构：不同 layer 的 cache 长度可能不同。
            # - sliding window / chunked attention 层：长度可能被上限卡住，出现 real_len < total_len（正常）
            # - full attention 层：通常会随 total_len 增长
            # 因此这里采样多个 layer 的 seq_length，用区间判断“是否明显不可能”。
            n_layers = int(getattr(self.model.config, "num_hidden_layers", 0) or 0)
            probe_idxs = {0, 1, n_layers - 1}
            for li in sorted(i for i in probe_idxs if 0 <= i < n_layers):
                try:
                    real_lens.append(int(self.cache.get_seq_length(layer_idx=li)))  # type: ignore[call-arg]
                except TypeError:
                    # 兼容不支持 layer_idx 的实现：退回到默认 layer
                    try:
                        real_lens = [int(self.cache.get_seq_length())]
                    except Exception:
                        real_lens = []
                    break
                except Exception:
                    continue

            if not real_lens:
                try:
                    real_lens = [int(self.cache.get_seq_length())]
                except Exception:
                    real_lens = []

        bad_real_len = False
        if real_lens:
            mx = max(real_lens)
            # 允许 mx < total_len（滑窗层会卡住），但不允许任何探针层 > total_len
            if mx > self.total_len:
                bad_real_len = True
            # 若历史已有前缀，却所有探针层都为 0，也很可疑
            if old_prefix_len > 0 and mx == 0:
                bad_real_len = True

        if (
            self.cache is None
            or self.total_len != old_prefix_len
            or bad_real_len
        ):
            reset_reason = "uninitialized_or_invariant_broken"
            logger.debug(
                "KvCacheManager: resetting cache (cache is %s, total_len=%d, "
                "prefix_len=%d, real_lens=%s)",
                "None" if self.cache is None else "set",
                self.total_len,
                old_prefix_len,
                str(real_lens),
            )
            self.reset()
            can_try_reuse = False

        reused_prefix_len = 0  # 实际复用的前缀长度

        # --- 3) 如果 cache 状态正常，再看“prompt 是否完全相同 / 是否中途分叉” ---
        if can_try_reuse and self.prefix_ids and input_ids == self.prefix_ids:
            # 为了拿到 fresh logits，完全相同的 prompt 也强制重建
            reset_reason = "prompt_unchanged_rebuild"
            logger.debug("KvCacheManager: prompt unchanged; rebuilding cache for fresh logits")
            self.reset()
            can_try_reuse = False

        if can_try_reuse:
            # 此时 self.prefix_ids / self.total_len 仍然是“上一轮完整序列”
            if lcp < old_prefix_len:
                reset_reason = f"prompt_diverged_in_prefix_lcp_{lcp}"
                logger.debug(
                    "KvCacheManager: prompt diverged inside old prefix (lcp=%d < old_len=%d); rebuilding cache",
                    lcp,
                    old_prefix_len,
                )
                self.reset()
                cur_len = 0
            else:
                # 完全前缀扩展：可以从 old_prefix_len 开始只算后缀
                cur_len = old_prefix_len
                reused_prefix_len = cur_len
        else:
            cur_len = 0

        # 现在要么 cache 是刚 reset 的（prefix_len=0, total_len=0），要么保持着旧前缀
        assert self.total_len == len(self.prefix_ids), (
            f"Invariant broken inside prefill_to: total_len={self.total_len}, "
            f"len(prefix_ids)={len(self.prefix_ids)}"
        )

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device  # 兼容 device_map="auto"

        idx = cur_len
        last_logits: torch.Tensor | None = None

        # --- 4) 对新增部分做 chunked prefill ---
        while idx < len(input_ids):
            chunk_ids = input_ids[idx: min(len(input_ids), idx + self.prefill_chunk)]
            chunk_len = len(chunk_ids)

            # 注意这里的 cache_position / attention_mask 用法，严格按官方文档推荐：
            cache_pos = torch.arange(
                self.total_len,
                self.total_len + chunk_len,
                dtype=torch.long,
                device=device,
            )
            attn_mask = torch.ones(
                1,
                self.total_len + chunk_len,
                dtype=torch.long,
                device=device,
            )

            inp = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            outputs = self.model(
                input_ids=inp,
                attention_mask=attn_mask,
                cache_position=cache_pos,
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
            )
            self.cache = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]

            self.total_len += chunk_len
            idx += chunk_len
            torch.cuda.empty_cache()

        # 更新 prefix_ids 为“本轮完整 prompt 序列”
        self.prefix_ids = list(input_ids)
        assert last_logits is not None, "prefill_to did not process any tokens"

        new_len = len(self.prefix_ids)
        forward_tokens = new_len - reused_prefix_len

        # --- 5) 累计统计信息 ---
        self.stats["prefill_calls"] += 1
        self.stats["prefill_tokens_total"] += new_len
        self.stats["prefill_tokens_forwarded"] += forward_tokens
        self.stats["prefill_tokens_reused"] += reused_prefix_len

        logger.debug(
            (
                "KV prefill: old_len=%d, new_len=%d, lcp=%d, "
                "reused_prefix_len=%d, forward_new_tokens=%d, "
                "reset_reason=%s"
            ),
            old_prefix_len,
            new_len,
            lcp,
            reused_prefix_len,
            forward_tokens,
            reset_reason,
        )

        return last_logits[0]

    def report_stats(self) -> dict:
        """
        返回当前 KV 使用统计（一个浅拷贝，方便外面打印 / JSON 化）。
        """
        return dict(self.stats)


# =========================
# 5) 运行时初始化
# =========================


def setup_runtime(_model_id: Optional[str] = None) -> None:
    """
    惰性初始化分词器、模型、Harmony 编码器与工具。
    """

    global tokenizer, model, encoding, browser_tool, python_tool, model_id, cfg, _decode_token_bytes, _assistant_start_token_ids

    cfg = _load_config()
    if cfg.debug:
        logger.setLevel(logging.DEBUG)

    if _model_id:
        model_id = _model_id

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # 为 Harmony 特殊 token 兜底注册（理论上已经在 vocab 内）
    specials = ["<|return|>", "<|call|>"]
    to_add = [t for t in specials if tokenizer.convert_tokens_to_ids(t) is None]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    # --- model ---
    # 使用 torch_dtype="auto" + device_map="auto"，让 Transformers 按 GPU / MXFP4 能力自动选择
    dtype = "auto"
    device_map: Any = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,       # 使用 torch_dtype 参数以兼容多数 Transformers 版本
        device_map=device_map,
    ).eval()

    # 若手动扩展了 vocab，需要 resize；失败可以忽略
    if to_add:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            logger.warning("resize_token_embeddings failed; continuing anyway")

    # --- Harmony 编码器 ---
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    _decode_token_bytes = _build_token_byte_decoder()
    # `<|start|>assistant` header（用于 completion prompt 末尾）；从空对话渲染即可得到稳定 token 序列。
    try:
        _assistant_start_token_ids = encoding.render_conversation_for_completion(
            Conversation.from_messages([]),
            Role.ASSISTANT,
        )
    except Exception:
        _assistant_start_token_ids = [200006, 173781]  # `<|start|>`, `assistant`

    # --- Tools: Browser / Python ---
    exa_key = os.getenv("EXA_API_KEY")
    if exa_key and exa_key.strip():
        try:
            backend = ExaBackend(source="web")
            browser_tool = SimpleBrowserTool(backend=backend)
        except Exception as e:
            logger.warning("Browser tool init failed; disabling browser. %s", e)
            browser_tool = None
    else:
        logger.info("EXA_API_KEY not set; browser tool disabled.")
        browser_tool = None

    py_impl = os.getenv("PYTHON_TOOL", "jupyter").strip().lower()
    use_docker_default = (py_impl in ("", "default", "docker"))
    if use_docker_default and DockerPythonTool is not None:
        try:
            python_tool = DockerPythonTool()  # type: ignore[call-arg]
        except Exception:
            python_tool = JupyterPythonTool()
    elif py_impl == "jupyter":
        python_tool = JupyterPythonTool()
    else:
        python_tool = JupyterPythonTool()

    if cfg.seed is not None:
        try:
            torch.manual_seed(cfg.seed)
        except Exception:
            pass

    # 进程退出时关闭 python kernel
    def _cleanup() -> None:
        try:
            if hasattr(python_tool, "shutdown"):
                python_tool.shutdown()
        except Exception:
            pass

    try:
        atexit.register(_cleanup)
    except Exception:
        pass


# =========================
# 6) 协程 / 工具调用 收集器
# =========================

def _run_coro_now(coro):
    """
    在“可能已有事件循环”的环境（如 Jupyter）里安全执行协程。
    """
    try:
        asyncio.get_running_loop()
        box: dict = {}

        def _th():
            nl = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(nl)
                box["val"] = nl.run_until_complete(coro)
            finally:
                nl.close()

        t = threading.Thread(target=_th, daemon=True)
        t.start()
        t.join()
        return box.get("val", None)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def collect_tool_messages(result) -> List[Message]:
    """把 python_tool.process(...) 或 browser_tool.process(...) 的返回统一为 list[Message]."""
    if result is None:
        return []
    if isinstance(result, list):
        return result
    if inspect.isasyncgen(result):
        async def _collect(agen):
            return [m async for m in agen]
        return _run_coro_now(_collect(result))
    if inspect.isawaitable(result):
        return collect_tool_messages(_run_coro_now(result))
    return [result]


# =========================
# 7) Harmony 采样循环（单次 assistant 生成）
# =========================

def generate_assistant_once(
    messages: List[Message],
    kv_manager: KvCacheManager,
    *,
    stream: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
    max_new_tokens: int = 32768,
) -> List[Message]:
    """
    对当前 conversation（messages）执行一次 assistant 生成，直到：
      - 采样到 Harmony stop token（如 <|return|> 或 <|call|>）
      - 或达到 max_new_tokens

    返回：encoding.parse_messages_from_completion_tokens 得到的“新 assistant 消息列表”。
    """
    assert model is not None and tokenizer is not None and encoding is not None
    assert _decode_token_bytes is not None

    # 1) 构造 Harmony prompt（优先使用“真实 token 历史”以保持 token 级一致性）
    if not kv_manager.history_ids:
        # 仅用于首次初始化：后续历史以“模型实际生成的 token”+“我们追加的 tool 消息 token”为准
        convo = Conversation.from_messages(messages)
        kv_manager.history_ids = encoding.render_conversation(convo)

    assert _assistant_start_token_ids is not None
    input_token_ids: List[int] = list(kv_manager.history_ids) + list(_assistant_start_token_ids)

    # 2) prefill：对完整 prompt 做前向，使用 prefix cache 复用
    with torch.inference_mode():
        last_logits = kv_manager.prefill_to(input_token_ids)

    # ---- decode 在同一个 DynamicCache 上“续写”KV ----
    # 注意：decode_cache 与 kv_manager.cache 指向同一个 DynamicCache 实例；
    # decode 过程中会在这个共享 cache 上继续追加 KV，decode 结束后我们只需
    # 同步 meta 信息（total_len / prefix_ids / stats）即可。
    decode_cache = kv_manager.cache
    decode_len = kv_manager.total_len

    # Harmony stop tokens（一般包含 <|return|> & <|call|>）
    stop_token_ids: List[int] = encoding.stop_tokens_for_assistant_actions()
    temperature = cfg.temperature if cfg is not None else 1.0
    top_p = cfg.top_p if cfg is not None else 1.0
    release_every = cfg.decode_release_every if cfg is not None else 256
    aggressive = bool(cfg.aggressive_empty_cache) if cfg is not None else True

    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    completion_ids: List[int] = []

    # 3) decode 循环：第一步使用 prefill 的 last_logits
    step = 0
    for _ in range(max_new_tokens):
        next_id = _softmax_sample_top_p(last_logits, temperature=temperature, top_p=top_p)

        is_stop = next_id in stop_token_ids

        # stop token 也要写入 completion_ids 并回灌模型（保证与 Harmony render 的 token 流一致，
        # 尤其是 tool-call 末尾的 <|call|>），但不作为可见输出流。
        completion_ids.append(next_id)

        if stream and not is_stop:
            token_bytes = _decode_token_bytes(next_id)
            decoded = utf8_decoder.decode(token_bytes, final=False)
            if decoded:
                if stream_callback is not None:
                    stream_callback(decoded)
                else:
                    sys.stdout.write(decoded)
                    sys.stdout.flush()

        # 把新 token 回灌模型，并更新 decode 阶段的 KV / 长度
        with torch.inference_mode():
            device = model.device
            # 本次新 token 的逻辑位置 = 当前 decode_len
            cache_pos = torch.tensor(
                [decode_len],
                dtype=torch.long,
                device=device,
            )
            # 全序列长度 = 过去 decode_len 个 token + 新 token
            seq_len = decode_len + 1
            attn_mask = torch.ones(
                1,
                seq_len,
                dtype=torch.long,
                device=device,
            )
            inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=inp,
                attention_mask=attn_mask,
                cache_position=cache_pos,
                past_key_values=decode_cache,
                use_cache=True,
                return_dict=True,
            )

        decode_cache = outputs.past_key_values
        decode_len += 1

        last_logits = outputs.logits[0, -1, :]

        step += 1
        if release_every > 0 and step % release_every == 0 and torch.cuda.is_available() and aggressive:
            _release_cuda()

        if is_stop:
            break

    # flush UTF-8 decoder
    if stream:
        tail = utf8_decoder.decode(b"", final=True)
        if tail:
            if stream_callback is not None:
                stream_callback(tail)
            else:
                sys.stdout.write(tail)
                sys.stdout.flush()

    # 将 decode 期的 KV 状态写回 KvCacheManager，保持不变量
    # （DynamicCache 本身已在 decode 过程中被原地更新，这里只同步 meta）
    kv_manager.cache = decode_cache
    kv_manager.total_len = decode_len
    if completion_ids:
        kv_manager.prefix_ids = list(input_token_ids) + list(completion_ids)
        kv_manager.stats["decode_tokens"] += len(completion_ids)
        kv_manager.history_ids = list(kv_manager.prefix_ids)

    # 4) 解析生成的 completion token 为 Harmony 消息
    if not completion_ids:
        return []

    # Harmony 推荐不要把 stop tokens（如 <|return|> / <|call|>）喂给 parser。
    parse_ids = list(completion_ids)
    while parse_ids and parse_ids[-1] in stop_token_ids:
        parse_ids.pop()
    if not parse_ids:
        return []

    gen_msgs: List[Message] = encoding.parse_messages_from_completion_tokens(
        parse_ids, Role.ASSISTANT
    )

    # 生成阶段结束后的简单清理（可选）
    if torch.cuda.is_available() and aggressive:
        _release_cuda()

    return gen_msgs


# =========================
# 8) 顶层对话 / 工具循环
# =========================

MAX_STEPS = 256


def main() -> None:
    global tokenizer, model, encoding, python_tool, browser_tool

    if tokenizer is None or model is None or encoding is None or python_tool is None:
        setup_runtime()

    # ---- 构造 System / Developer / User 初始消息 ----
    system_msg = (
        SystemContent.new()
        .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        .with_knowledge_cutoff("2024-06")
        .with_conversation_start_date(str(date.today()))
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_required_channels(["analysis", "commentary", "final"])
    )
    if python_tool is not None:
        system_msg = system_msg.with_python_tool()
    if browser_tool is not None:
        system_msg = system_msg.with_browser_tool()

    developer_msg = (
        DeveloperContent.new().with_instructions(
            ""  # "You are a helpful assistant."
        )
    )

    user_msg = r"""
    计算下面级数的前 1000 项和，并保留 10 位小数，尽量使用 Python 工具来计算：
    S = sum_{n=1..1000} (-1)^{n+1} / (n^2 + n)
    """

    messages: List[Message] = [
        Message.from_role_and_content(Role.SYSTEM, system_msg),
        Message.from_role_and_content(Role.DEVELOPER, developer_msg),
        Message.from_role_and_content(Role.USER, user_msg),
    ]

    kv_manager = KvCacheManager(model, prefill_chunk=(cfg.prefill_chunk if cfg else 128))
    # 初始化 token 历史：从当前 messages 渲染（后续将用“真实 token 流”增量维护）
    kv_manager.history_ids = encoding.render_conversation(Conversation.from_messages(messages))

    for _ in range(MAX_STEPS):
        # 1) 让模型说话（可能是 tool 调用，也可能直接给出最终答案）
        gen_msgs = generate_assistant_once(
            messages,
            kv_manager,
            stream=True,
            stream_callback=None,
        )
        if not gen_msgs:
            # 没有新 token，继续下一轮尝试
            continue

        messages.extend(gen_msgs)
        last = gen_msgs[-1]

        # 2) 检查是否是工具调用
        name = (last.recipient or "").strip()
        if name:
            base = name.split(None, 1)[0].split(".", 1)[0]
            if base == "python":
                # 调用 python 工具
                py_call = coerce_python_call_message(last)
                try:
                    tool_out = python_tool.process(py_call)  # type: ignore[union-attr]
                    tool_msgs = collect_tool_messages(tool_out)
                except Exception as e:
                    err = json.dumps({"error": f"python tool failed: {e}"})
                    tool_msgs = [
                        Message.from_author_and_content(
                            Author.new(Role.TOOL, "python"),
                            err,
                        )
                        .with_channel("commentary")
                        .with_recipient("assistant")
                        .with_content_type("json")
                    ]
                messages.extend(tool_msgs)
                for m in tool_msgs:
                    kv_manager.history_ids.extend(encoding.render(m))
                # 下一轮循环会基于新的 messages 继续采样
                continue

            elif base == "browser":
                if browser_tool is None:
                    tool_msgs = [
                        Message.from_author_and_content(
                            Author.new(Role.TOOL, "browser"),
                            json.dumps({"error": "browser tool disabled or not configured"}),
                        )
                        .with_channel("commentary")
                        .with_recipient("assistant")
                        .with_content_type("json")
                    ]
                else:
                    try:
                        tool_out = browser_tool.process(last)
                        tool_msgs = collect_tool_messages(tool_out)
                    except Exception as e:
                        err = json.dumps({"error": f"browser tool failed: {e}"})
                        tool_msgs = [
                            Message.from_author_and_content(
                                Author.new(Role.TOOL, "browser"),
                                err,
                            )
                            .with_channel("commentary")
                            .with_recipient("assistant")
                            .with_content_type("json")
                        ]

                messages.extend(tool_msgs)
                for m in tool_msgs:
                    kv_manager.history_ids.extend(encoding.render(m))
                continue

            else:
                # 未知工具：回填错误，防止模型“以为工具调用成功”
                messages.append(
                    Message.from_author_and_content(
                        Author.new(Role.TOOL, name),
                        json.dumps({"error": f"Unknown tool '{name}'."}),
                    )
                    .with_channel("commentary")
                    .with_recipient("assistant")
                    .with_content_type("json")
                )
                kv_manager.history_ids.extend(encoding.render(messages[-1]))
                continue

        # 3) 若不是工具调用，则检查是否已经给出 final 消息
        chan = (last.channel or "").strip().lower()
        if chan == "final":
            break
        # 否则继续下一轮（例如模型还在 analysis channel 输出推理）


if __name__ == "__main__":
    main()
