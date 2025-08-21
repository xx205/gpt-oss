# -*- coding: utf-8 -*-
# pip install -U "torch>=2.8.0" "triton>=3.4.0" "transformers>=4.55.2" kernels accelerate openai-harmony gpt-oss jupyter_client ipykernel pyzmq

import asyncio
import inspect
import json
import sys
import threading
import gc
import time
import os

from collections.abc import Iterable
from datetime import date

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
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.cache_utils import DynamicCache as _HF_DynamicCache  # type: ignore
except Exception:  # noqa: BLE001
    _HF_DynamicCache = None
from gpt_oss.tools.simple_browser import SimpleBrowserTool, ExaBackend
# Prefer the repo-default Docker Python tool if available; fall back to Jupyter
try:
    # Stateless, non-REPL python tool executed in a docker sandbox
    from gpt_oss.tools.python_docker.docker_tool import PythonTool as DockerPythonTool  # type: ignore
except Exception:  # noqa: BLE001
    DockerPythonTool = None  # type: ignore

# =========================
# 0) 模型与编码
# =========================
model_id = "openai/gpt-oss-20b"

# 延迟初始化：避免 import 即触发大模型与 GPU 资源占用
tokenizer = None
model = None
encoding = None
browser_tool = None
python_tool = None
_pkv_debug_printed = False


# ==== Debug helpers (把 token ids 还原为原始文本，并打印 lcp 区段) ====
def _decode_tokens(ids: list[int]) -> str:
    # 不跳过特殊 token，避免把 <|return|> 之类的信息丢掉
    assert tokenizer is not None, "Tokenizer not initialized. Call setup_runtime() first."
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

def _show_lcp_debug(prev_ids: list[int], cur_ids: list[int], lcp_len: int, *, tail_tokens: int = 12, max_chars: int | None = 400) -> None:
    print(f"\n[DEBUG] LCP = {lcp_len} tokens")
    # 1) 打印 lcp 对应的原始文本（可选择只看末尾若干字符，避免刷屏）
    if lcp_len > 0:
        lcp_text = _decode_tokens(cur_ids[:lcp_len])
        if max_chars is not None and len(lcp_text) > max_chars:
            print("[LCP TEXT]", repr("…" + lcp_text[-max_chars:]))
        else:
            print("[LCP TEXT]", repr(lcp_text))
    else:
        print("[LCP TEXT] <empty> (no common prefix)")

    # 2) 再看分叉处两边的“第一小段”以便定位差异
    prev_tail_ids = prev_ids[lcp_len:lcp_len + tail_tokens]
    cur_tail_ids  = cur_ids[lcp_len:lcp_len + tail_tokens]
    if prev_tail_ids or cur_tail_ids:
        print("[DIVERGENCE @ LCP]")
        print("  prev ids:", prev_tail_ids)
        print("  curr ids:", cur_tail_ids)
        try:
            print("  prev tok:", tokenizer.convert_ids_to_tokens(prev_tail_ids))
            print("  curr tok:", tokenizer.convert_ids_to_tokens(cur_tail_ids))
        except Exception:
            pass
        print("  prev text:", repr(_decode_tokens(prev_tail_ids)))
        print("  curr text:", repr(_decode_tokens(cur_tail_ids)))

# ---- memory helpers: 丢引用后立刻回收 CUDA 缓存 ----
def _release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            torch.cuda.empty_cache()

# =========================
# 1) 工具（Python / Browser）（安全位留空供加固）
# =========================
# from gpt_oss.tools.python_docker.docker_tool import PythonTool


def _content_to_code_str(msg) -> str:
    """从 Harmony Message 中稳妥抽出代码字符串：兼容 JSON 包装与 code/text 片段。"""
    c = getattr(msg, "content", "")
    # 情况1：纯字符串（可能是 {"code": "..."} 的JSON文本）
    if isinstance(c, str):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and isinstance(obj.get("code"), str):
                return obj["code"]
        except Exception:
            return c
    # 情况2：内容片段列表（模型常用 to=python code 的形式）
    try:
        if isinstance(c, Iterable) and not isinstance(c, (bytes, bytearray)):
            parts = []
            for p in c:
                if hasattr(p, "code") and isinstance(p.code, str):
                    parts.append(p.code)
                elif hasattr(p, "text") and isinstance(p.text, str):
                    parts.append(p.text)
            if parts:
                return "\n".join(parts)
    except Exception:
        pass
    # 兜底
    t = getattr(c, "text", None)
    return t if isinstance(t, str) else str(c)

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
        self.kc = self.km.client()          # or .blocking_client()
        self.kc.start_channels()

    def shutdown(self):
        try:
            self.kc.stop_channels()
        finally:
            try:
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass

    def _run_code(self, code: str):
        msg_id = self.kc.execute(code, allow_stdin=False, stop_on_error=True)
        t0 = time.time()
        stdout_parts, stderr_parts, displays, last_text_result = [], [], [], None

        while True:
            if time.time() - t0 > self.timeout_s:
                stderr_parts.append(f"\n[Timeout] execution exceeded {self.timeout_s}s")
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
            "displays": displays,   # e.g. [{"mime":"image/png","data":"...base64..."}]
            "files": [],
        }

    def process(self, py_call_msg: Message):
        # 关键：把 Message 内容抽成 str，再执行
        code = _content_to_code_str(py_call_msg)

        payload = self._run_code(code)

        # ✅ 正确的“工具回传”形状：作者=具体工具名；收件人=assistant；走 commentary
        tool_msg = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, "python"),
                json.dumps(payload)
            )
            .with_channel("commentary")
            .with_recipient("assistant")
            .with_content_type("json")
        )
        return [tool_msg]

def setup_runtime(_model_id: str | None = None) -> None:
    """惰性初始化分词器、模型、编码与工具。

    放入 main() 调用，避免 import 时执行重活；也便于后续
    通过参数覆盖 model_id。
    """
    global tokenizer, model, encoding, browser_tool, python_tool, model_id
    # 允许通过函数参数或环境变量覆盖模型
    if _model_id:
        model_id = _model_id

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # —— 注册 Harmony 特殊 token（若缺失）——
    _SPECIALS = ["<|return|>", "<|call|>"]
    to_add = [t for t in _SPECIALS if tokenizer.convert_tokens_to_ids(t) is None]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    # 模型（按环境选择设备）
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        # attn_implementation="sdpa",
    )

    # 若刚注册过特殊 token，这里补一次 resize（失败忽略）
    if to_add:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    # 编码
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # 工具（Browser / Python）
    backend = ExaBackend(source="web")  # 需设置 EXA_API_KEY
    browser_tool = SimpleBrowserTool(backend=backend)
    # 允许通过环境变量切换 python 工具实现：docker | jupyter
    py_impl = os.getenv("HARMONY_PYTHON_TOOL", "jupyter").strip().lower()
    use_docker_default = (py_impl in ("", "default", "docker"))
    if use_docker_default and DockerPythonTool is not None:
        try:
            python_tool = DockerPythonTool()  # type: ignore[call-arg]
            # print("[python tool] using docker-based PythonTool")
        except Exception:
            python_tool = JupyterPythonTool()
            # print("[python tool] docker init failed; falling back to Jupyter kernel")
    elif py_impl == "jupyter":
        python_tool = JupyterPythonTool()
        # print("[python tool] using Jupyter kernel tool")
    else:
        # 无法导入 docker 实现时的兜底
        python_tool = JupyterPythonTool()
        # if DockerPythonTool is None:
        #     print("[python tool] docker tool not found; using Jupyter kernel")
        # else:
        #     print("[python tool] unknown HARMONY_PYTHON_TOOL; using Jupyter kernel")

    # 写回全局（避免局部变量遮蔽）
    globals()["browser_tool"] = browser_tool
    globals()["python_tool"] = python_tool

# =========================
# 2) System / Developer / User
# =========================
system_msg = (
    SystemContent.new()
    .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
    .with_knowledge_cutoff("2024-06")
    .with_conversation_start_date(str(date.today()))
    .with_reasoning_effort(ReasoningEffort.MEDIUM)
    .with_required_channels(["analysis", "commentary", "final"])
    .with_python_tool()  # 使用内置的（训练时）python 工具定义 = 有状态 Jupyter 约定
    # .with_browser_tool()
)

developer_msg = (
    DeveloperContent.new().with_instructions(
        """
        """
    )
)

user_msg = r"""
计算下面级数的前 1000 项和，并保留 10 位小数：
S = sum_{n=1..1000} (-1)^{n+1} / (n^2 + n)
"""

# =========================
# 3) 终止 token（稳健获取）
# =========================
def _safe_token_id(tok):
    tid = tokenizer.convert_tokens_to_ids(tok)
    return tid if isinstance(tid, int) and tid >= 0 else None

def _collect_eos_ids():
    ids = []
    for sym in ["<|return|>", "<|call|>"]:
        tid = _safe_token_id(sym)
        if tid is not None:
            ids.append(tid)
    try:
        if getattr(tokenizer, "eos_token", None):
            tid = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            if isinstance(tid, int):
                ids.append(tid)
    except Exception:
        pass
    return list(dict.fromkeys(ids))

# =========================
# 4) 采样 / 规整 / 收敛
# =========================
def generate_once(
    msgs,
    *,
    stream=True,
    max_new_tokens=32768,
    past_key_values=None,
    processed_tokens=0,
):
    """Generate one step using cached KV state (manual streaming decode).

    Args:
        msgs: current conversation messages.
        stream: whether to stream decoded text.
        max_new_tokens: generation limit.
        past_key_values: previous KV cache from the model.
        processed_tokens: number of tokens already seen by the model.

    Returns:
        out_ids: newly generated token ids.
        past_key_values: updated KV cache.
        processed_tokens: updated token count including model output.
    """

    # ---- helpers ----
    def _to_dev(ids_list: list[int]) -> torch.Tensor:
        return torch.tensor([ids_list], dtype=torch.long, device=model.device)

    def _softmax_sample_top_p(logits: torch.Tensor, *, temperature: float = 1.0, top_p: float = 1.0) -> int:
        # logits: [vocab]
        import torch.nn.functional as F
        if temperature <= 0:
            temperature = 1e-6
        logits = (logits / temperature).to(torch.float32)
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf > top_p
            # ensure at least 1 token kept
            if mask[0]:
                mask[0] = False
            sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
            sorted_probs = sorted_probs / sorted_probs.sum()
            next_id = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[next_id].item())
        else:
            return int(torch.multinomial(probs, num_samples=1).item())

    # ---- render prompt ----
    convo = Conversation.from_messages(msgs)
    input_token_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    total_len = len(input_token_ids)
    new_input_ids = input_token_ids[processed_tokens:]

    # # 观测：生成调用前的 prompt 与 KV 情况
    # try:
    #     kv_len_before = _kv_seq_len(past_key_values)
    #     print(f"[GEN] before: prompt_total={total_len}, delta={len(new_input_ids)}, processed={processed_tokens}, kv_len={kv_len_before}, gpu=({_gpu_mem_stats()})")
    # except Exception:
    #     pass

    # ---- prefill to build/extend KV up to last prompt token ----
    prefill_upto = max(total_len - 1, 0)
    out_ids: list[int] = []
    with torch.inference_mode():
        # extend PKV with any new input ids except the last token
        if processed_tokens < prefill_upto:
            # feed in chunks to reduce peak
            chunk = int(os.getenv("HARMONY_PREFILL_CHUNK", "1024"))
            aggressive = os.getenv("HARMONY_AGGRESSIVE_EMPTY_CACHE", "1").lower() in {"1","true","yes"}
            i = processed_tokens
            while i < prefill_upto:
                j = min(prefill_upto, i + chunk)
                ids = _to_dev(input_token_ids[i:j])
                outputs = model(input_ids=ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
                past_key_values = outputs.past_key_values
                i = j
                # aggressively release transient activation if requested
                try:
                    del outputs
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    if aggressive:
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except Exception:
                            torch.cuda.empty_cache()
            processed_tokens = prefill_upto

        # choose the starter token (the last prompt token) if exists, else fallback to BOS
        if total_len > 0:
            cur_token = input_token_ids[-1]
        else:
            # try to use tokenizer.bos_token_id, else 0
            cur_token = getattr(tokenizer, 'bos_token_id', 0) or 0

        # streaming decode loop
        eids = _collect_eos_ids()
        temperature = 1.0
        top_p = 1.0
        aggressive = os.getenv("HARMONY_AGGRESSIVE_EMPTY_CACHE", "1").lower() in {"1","true","yes"}
        release_every = int(os.getenv("HARMONY_DECODE_RELEASE_EVERY", "256"))
        step = 0
        for _ in range(max_new_tokens):
            inp = _to_dev([cur_token])
            outputs = model(input_ids=inp, past_key_values=past_key_values, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            next_id = _softmax_sample_top_p(logits, temperature=temperature, top_p=top_p)
            out_ids.append(next_id)

            # stream token
            try:
                sys.stdout.write(tokenizer.decode([next_id], skip_special_tokens=False, clean_up_tokenization_spaces=False))
                sys.stdout.flush()
            except Exception:
                pass

            if eids and next_id in eids:
                break
            cur_token = next_id
            # periodic aggressive release during decode
            step += 1
            if (step % max(1, release_every)) == 0 and torch.cuda.is_available() and aggressive:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    torch.cuda.empty_cache()

    # 同步 + 清理缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _release_cuda()

    processed_tokens = total_len + len(out_ids)

    # 观测：生成调用后的 KV 情况
    # try:
    #     kv_len_after = _kv_seq_len(past_key_values)
    #     print(f"[GEN] after: out_tokens={len(out_ids)}, processed={processed_tokens}, kv_len={kv_len_after}, gpu=({_gpu_mem_stats()})")
    # except Exception:
    #     pass
    return out_ids, past_key_values, processed_tokens

def _run_coro_now(coro):
    """
    在可能“已有事件循环运行中”的环境（如 Jupyter/异步服务）里安全执行协程：
    若检测到已有 running loop，则转到新线程里创建独立 loop 执行。
    """
    try:
        asyncio.get_running_loop()  # 若无运行中 loop 会抛 RuntimeError
        box = {}

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

def collect_tool_messages(result):
    """把 python_tool.process(...) 的返回统一为 list[Message]。"""
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

def _content_to_text(content) -> str:
    """把 Harmony 内容统一成纯文本（处理 TextContent 列表等）"""
    if isinstance(content, str):
        return content
    try:
        if isinstance(content, Iterable) and not isinstance(content, (bytes, bytearray)):
            parts = []
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

def coerce_python_call_message(last_msg: Message) -> Message:
    """
    把“模型发起的工具调用”规范化为 PythonTool 可执行的格式。

    模型可能传回 JSON 如 {"code": "print(2+2)"}，
    而 PythonTool 期望是纯 Python 源码。这里解析 JSON，
    并返回只包含代码字符串的消息。
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


def _longest_common_prefix(a: list[int], b: list[int]) -> int:
    """Return length of longest common prefix of two token lists."""
    l = 0
    for x, y in zip(a, b):
        if x != y:
            break
        l += 1
    return l


def _truncate_past_key_values(past_key_values, length: int):
    """返回一份被“物理紧凑化”的 KV（新对象）。失败则返回 None。"""
    if past_key_values is None:
        return None

    # 1) DynamicCache 兼容：尽量抽取出底层 K/V 并小拷贝
    def _compact_dynamic_cache(pkv, L):
        # 让逻辑长度先生效（万一模型内部用到）
        try:
            if hasattr(pkv, "truncate"):
                pkv.truncate(L)
                # 若已支持逻辑截断，但无法物理提取底层张量，也可以直接返回 pkv 以便继续复用
                # 先尝试物理紧凑化，失败再回退
        except Exception:
            pass

        # 0) DynamicCache layers[].keys/values → 构建新的 DynamicCache
        try:
            layers = getattr(pkv, "layers", None)
            if _HF_DynamicCache is not None and isinstance(layers, (list, tuple)) and layers:
                new_cache = _HF_DynamicCache()
                ok = True
                for idx, layer in enumerate(layers):
                    k = getattr(layer, "keys", None)
                    v = getattr(layer, "values", None)
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        new_k = k[..., :L, :].contiguous().clone()
                        new_v = v[..., :L, :].contiguous().clone()
                        new_cache.update(new_k, new_v, idx)
                    else:
                        ok = False
                        break
                if ok:
                    return new_cache
        except Exception:
            pass

        # 顶层 key_cache/value_cache 直接切片（可能已弃用，但尽量利用）
        try:
            kc = getattr(pkv, "key_cache", None)
            vc = getattr(pkv, "value_cache", None)
            if _HF_DynamicCache is not None and isinstance(kc, (list, tuple)) and isinstance(vc, (list, tuple)) and len(kc) == len(vc) and kc:
                new_cache = _HF_DynamicCache()
                ok = True
                for idx, (k, v) in enumerate(zip(kc, vc)):
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        new_k = k[..., :L, :].contiguous().clone()
                        new_v = v[..., :L, :].contiguous().clone()
                        new_cache.update(new_k, new_v, idx)
                    else:
                        ok = False
                        break
                if ok:
                    return new_cache
        except Exception:
            pass

        # 通过 to_legacy_cache() 转换为旧格式再切片
        try:
            if hasattr(pkv, "to_legacy_cache"):
                legacy = pkv.to_legacy_cache()
                if _HF_DynamicCache is not None and isinstance(legacy, (list, tuple)) and legacy:
                    new_cache = _HF_DynamicCache()
                    ok = True
                    for idx, layer in enumerate(legacy):
                        if isinstance(layer, (list, tuple)) and len(layer) == 2:
                            k, v = layer
                            if torch.is_tensor(k) and torch.is_tensor(v):
                                new_k = k[..., :L, :].contiguous().clone()
                                new_v = v[..., :L, :].contiguous().clone()
                                new_cache.update(new_k, new_v, idx)
                            else:
                                ok = False
                                break
                        else:
                            ok = False
                            break
                    if ok:
                        return new_cache
        except Exception:
            pass

        # 找到底层缓存并小拷贝为 list[(k,v)]
        for attr in ("caches", "layers", "kv_cache", "key_value_caches"):
            if hasattr(pkv, attr):
                caches = getattr(pkv, attr)
                new_list = []
                for layer in caches:
                    # 兼容不同字段命名
                    k = (
                        getattr(layer, "key_cache", None)
                        or getattr(layer, "k_cache", None)
                        or getattr(layer, "k", None)
                    )
                    v = (
                        getattr(layer, "value_cache", None)
                        or getattr(layer, "v_cache", None)
                        or getattr(layer, "v", None)
                    )
                    # 若 layer 本身就是 (k,v) 形式
                    if (k is None or v is None) and isinstance(layer, (list, tuple)) and len(layer) == 2:
                        k, v = layer
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        new_k = k[..., :L, :].contiguous().clone()
                        new_v = v[..., :L, :].contiguous().clone()
                        new_list.append((new_k, new_v))
                    else:
                        return None
                return new_list
        # 无法物理提取时，若已做过逻辑 truncate，则直接返回 pkv（继续复用，而非丢弃）
        try:
            if hasattr(pkv, "truncate"):
                return pkv
        except Exception:
            pass
        return None

    # 2) 传统 list/tuple 形式：小拷贝
    if isinstance(past_key_values, (list, tuple)):
        new_list = []
        for layer in past_key_values:
            if not (isinstance(layer, (list, tuple)) and len(layer) == 2):
                return None
            k, v = layer
            if not (torch.is_tensor(k) and torch.is_tensor(v)):
                return None
            new_k = k[..., :length, :].contiguous().clone()
            new_v = v[..., :length, :].contiguous().clone()
            new_list.append((new_k, new_v))
        return new_list

    # 3) DynamicCache 或其它结构：尝试紧凑化为 list[(k,v)]
    compact = _compact_dynamic_cache(past_key_values, length)
    if compact is not None:
        return compact

    return None

def _rebuild_pkv_for_prefix(input_token_ids: list[int], upto: int):
    """Rebuild a fresh KV cache for the given prefix [0:upto] using chunked prefill.

    This is a safe fallback when physical truncation is unavailable.
    """
    print("-------------------- rebuild happened -------------------------")
    if upto <= 0:
        return None
    import os
    pkv = None
    chunk = int(os.getenv("HARMONY_PREFILL_CHUNK", "1024"))
    aggressive = os.getenv("HARMONY_AGGRESSIVE_EMPTY_CACHE", "1").lower() in {"1","true","yes"}
    with torch.inference_mode():
        i = 0
        while i < upto:
            j = min(upto, i + chunk)
            ids = torch.tensor([input_token_ids[i:j]], dtype=torch.long, device=model.device)
            outputs = model(input_ids=ids, past_key_values=pkv, use_cache=True, return_dict=True)
            pkv = outputs.past_key_values
            i = j
            # aggressively release transient activation if requested
            try:
                del outputs
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if aggressive:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        torch.cuda.empty_cache()
    return pkv

def _extend_pkv_with_tokens(past_key_values, token_ids: list[int]):
    """Extend existing PKV by feeding the given token_ids (in chunks)."""
    import os
    if not token_ids:
        return past_key_values
    chunk = int(os.getenv("HARMONY_PREFILL_CHUNK", "1024"))
    aggressive = os.getenv("HARMONY_AGGRESSIVE_EMPTY_CACHE", "0").lower() in {"1","true","yes"}
    with torch.inference_mode():
        i = 0
        while i < len(token_ids):
            j = min(len(token_ids), i + chunk)
            ids = torch.tensor([token_ids[i:j]], dtype=torch.long, device=model.device)
            outputs = model(input_ids=ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            i = j
            try:
                del outputs
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if aggressive:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        torch.cuda.empty_cache()
    return past_key_values

def _debug_pkv_once(pkv, tag: str = ""):
    global _pkv_debug_printed
    if _pkv_debug_printed or pkv is None:
        return
    _pkv_debug_printed = True
    try:
        tname = type(pkv).__name__
        has_trunc = hasattr(pkv, "truncate")
        has_getlen = hasattr(pkv, "get_seq_length")
        layer_info = {}
        for attr in ("caches", "layers", "kv_cache", "key_value_caches"):
            if hasattr(pkv, attr):
                caches = getattr(pkv, attr)
                if caches:
                    layer0 = caches[0]
                    fields = {}
                    for k in ("key_cache","k_cache","k","value_cache","v_cache","v"):
                        fields[k] = hasattr(layer0, k)
                    # 若 layer0 是二元组
                    if isinstance(layer0, (list, tuple)) and len(layer0) == 2:
                        fields["tuple_like"] = True
                    layer_info[attr] = fields
                break
        # print(f"[PKV-DEBUG] tag={tag}, type={tname}, has_truncate={has_trunc}, has_get_seq_length={has_getlen}, layer_fields={layer_info}")
    except Exception:
        pass

# =========================
# 4.1) 观测：KV 缓存长度探测
# =========================
def _kv_seq_len(past_key_values) -> int:
    """Best-effort 获取 KV 缓存中的序列长度（不分配、不拷贝）。

    返回 0 表示当前没有可用 KV（如首次生成或被清空）。
    """
    if past_key_values is None:
        return 0

    # list/tuple 形式: [(k, v), ...]
    if isinstance(past_key_values, (list, tuple)) and past_key_values:
        first = past_key_values[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            k = first[0]
            try:
                if torch.is_tensor(k) and k.ndim >= 2:
                    return int(k.shape[-2])
            except Exception:
                pass
        return 0

    # 直接方法: 某些缓存对象提供 get_seq_length()
    try:
        if hasattr(past_key_values, "get_seq_length"):
            v = past_key_values.get_seq_length()
            if isinstance(v, int):
                return v
    except Exception:
        pass

    # 动态缓存对象: 优先通过 layers[].keys 获取
    try:
        layers = getattr(past_key_values, "layers", None)
        if isinstance(layers, (list, tuple)) and layers:
            layer0 = layers[0]
            k = getattr(layer0, "keys", None)
            if torch.is_tensor(k) and k.ndim >= 2:
                seq_len = int(k.shape[-2])
                if seq_len <= 0 and k.ndim >= 3:
                    seq_len = int(k.shape[-3])
                return seq_len
    except Exception:
        pass

    # 动态缓存对象: 尝试访问底层 caches/layers/kv_cache/key_value_caches
    for attr in ("caches", "layers", "kv_cache", "key_value_caches"):
        if hasattr(past_key_values, attr):
            caches = getattr(past_key_values, attr)
            try:
                if caches:
                    layer0 = caches[0]
                    k = (
                        getattr(layer0, "key_cache", None)
                        or getattr(layer0, "k_cache", None)
                        or getattr(layer0, "k", None)
                    )
                    if torch.is_tensor(k) and k.ndim >= 2:
                        seq_len = int(k.shape[-2])
                        if seq_len <= 0 and k.ndim >= 3:
                            seq_len = int(k.shape[-3])
                        return seq_len
            except Exception:
                pass
            break

    # DynamicCache 顶层 key_cache/value_cache 可用
    try:
        kc = getattr(past_key_values, "key_cache", None)
        if isinstance(kc, (list, tuple)) and kc:
            k0 = kc[0]
            if torch.is_tensor(k0) and k0.ndim >= 2:
                seq_len = int(k0.shape[-2])
                if seq_len <= 0 and k0.ndim >= 3:
                    seq_len = int(k0.shape[-3])
                return seq_len
    except Exception:
        pass

    return 0

def _gpu_mem_stats() -> str:
    """返回当前 CUDA 显存简要统计字符串。"""
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    try:
        alloc = torch.cuda.memory_allocated()
        reserv = torch.cuda.memory_reserved()
        max_reserv = torch.cuda.max_memory_reserved()
        return f"alloc={alloc/1e9:.2f}GB, reserved={reserv/1e9:.2f}GB, max_reserved={max_reserv/1e9:.2f}GB"
    except Exception:
        return "cuda_stats_error"

# =========================
# 5) 工具循环
# =========================
MAX_STEPS = 64


def _assert_tools_named(sysmsg):
    tools_ns = getattr(sysmsg, "tools", None)
    assert tools_ns, "No tools declared on system_msg"

    bad_ns = []
    bad_funcs = []
    summary = []

    for key, spec in tools_ns.items():
        # spec 是 Pydantic 模型（ToolNamespaceConfig）
        name = getattr(spec, "name", None)
        if not name and hasattr(spec, "model_dump"):
            name = spec.model_dump().get("name")

        # 记录概况
        summary.append((key, name, type(spec).__name__))

        if not name or not str(name).strip():
            bad_ns.append(key)

        # 逐个检查该命名空间下的函数（如 browser.search/open/find）
        funcs = getattr(spec, "tools", []) or []
        for i, f in enumerate(funcs):
            fname = getattr(f, "name", None)
            if not fname and hasattr(f, "model_dump"):
                fname = f.model_dump().get("name")
            if not fname or not str(fname).strip():
                bad_funcs.append(f"{key}[{i}]")

    print("tool namespaces:", summary)
    assert not bad_ns, f"Unnamed tool namespaces: {bad_ns}"
    assert not bad_funcs, f"Unnamed functions in namespaces: {bad_funcs}"

# _assert_tools_named(system_msg)

def main() -> None:
    # 在执行前初始化运行时（模型、分词器、工具）
    if tokenizer is None or model is None or encoding is None or browser_tool is None or python_tool is None:
        setup_runtime()
    messages = [
        Message.from_role_and_content(Role.SYSTEM, system_msg),
        Message.from_role_and_content(Role.DEVELOPER, developer_msg),
        Message.from_role_and_content(Role.USER, user_msg),
    ]

    past_key_values = None
    processed_tokens = 0
    last_seen_ids = None

    for _ in range(MAX_STEPS):
        convo = Conversation.from_messages(messages)
        cur_prefix_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        if last_seen_ids is not None and past_key_values is not None:
            if last_seen_ids != cur_prefix_ids:
                # 计算 lcp，并打印可读调试信息
                lcp = _longest_common_prefix(last_seen_ids, cur_prefix_ids)
                # _show_lcp_debug(last_seen_ids, cur_prefix_ids, lcp, tail_tokens=12, max_chars=400)

                if lcp == 0:
                    # 完全不同前缀：无法复用，清理旧引用
                    old_pkv = past_key_values
                    past_key_values = None
                    processed_tokens = 0
                    last_seen_ids = None
                    try:
                        del old_pkv
                    except Exception:
                        pass
                    _release_cuda()
                else:
                    # 复用到 lcp，优先尝试物理紧凑化；失败则重建至 lcp
                    old_pkv = past_key_values
                    _debug_pkv_once(old_pkv, tag="lcp_path")
                    truncated = _truncate_past_key_values(old_pkv, lcp)
                    if truncated is None:
                        # 安全回退：按当前前缀重建 PKV 到 lcp
                        rebuilt = _rebuild_pkv_for_prefix(cur_prefix_ids, lcp)
                        if rebuilt is not None:
                            past_key_values = rebuilt
                            processed_tokens = lcp
                            last_seen_ids = cur_prefix_ids[:lcp]
                            # try:
                            #     print(f"[PKV] rebuild_to_lcp lcp={lcp}")
                            # except Exception:
                            #     pass
                        else:
                            past_key_values = None
                            processed_tokens = 0
                            last_seen_ids = None
                    else:
                        past_key_values = truncated
                        processed_tokens = lcp
                        last_seen_ids = cur_prefix_ids[:lcp]
                        # try:
                        #     print(f"[PKV] truncate_to_lcp lcp={lcp}")
                        # except Exception:
                        #     pass
                    # 丢掉旧引用并回收
                    try:
                        del old_pkv
                    except Exception:
                        pass
                    _release_cuda()

        prev_prefix_ids = cur_prefix_ids

        out_ids, past_key_values, processed_tokens = generate_once(
            messages,
            stream=True,
            past_key_values=past_key_values,
            processed_tokens=processed_tokens,
        )

        last_seen_ids = prev_prefix_ids + out_ids

        gen_msgs = encoding.parse_messages_from_completion_tokens(out_ids, role=Role.ASSISTANT)
        messages.extend(gen_msgs)
        # 对齐到“规范化后的前缀”，最大化复用，尽量不回退
        try:
            canonical = encoding.render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)
            gen_concat = prev_prefix_ids + out_ids
            if canonical != gen_concat:
                lcp_r = _longest_common_prefix(gen_concat, canonical)
                # 尝试在已有 PKV 上截到 LCP，再仅补算 delta
                if past_key_values is not None and lcp_r > 0:
                    _debug_pkv_once(past_key_values, tag="post_norm_align")
                    pkv_trunc = _truncate_past_key_values(past_key_values, lcp_r)
                    if pkv_trunc is not None:
                        past_key_values = pkv_trunc
                        delta = canonical[lcp_r:]
                        if delta:
                            past_key_values = _extend_pkv_with_tokens(past_key_values, delta)
                        processed_tokens = len(canonical)
                        last_seen_ids = canonical
                    else:
                        # 截断不可用：直接重建到 canonical
                        rebuilt = _rebuild_pkv_for_prefix(canonical, len(canonical))
                        if rebuilt is not None:
                            past_key_values = rebuilt
                            processed_tokens = len(canonical)
                            last_seen_ids = canonical
                        else:
                            # 保守退避：维持原状（下一轮会通过现有逻辑处理）
                            pass
                else:
                    # 无 PKV 或 LCP=0：直接重建到 canonical（避免下一轮回退）
                    rebuilt = _rebuild_pkv_for_prefix(canonical, len(canonical))
                    if rebuilt is not None:
                        past_key_values = rebuilt
                        processed_tokens = len(canonical)
                        last_seen_ids = canonical
        except Exception:
            pass

        last = gen_msgs[-1]
        name = (last.recipient or "").strip()

        if name:
            # 兼容 "python", "python.exec", "python code" 等变体
            base = name.split(None, 1)[0].split(".", 1)[0]
            if base == "python":
                # print(f"\n[TOOL DISPATCH] -> {name}")
                try:
                    before_len = _kv_seq_len(past_key_values)
                except Exception:
                    before_len = -1
                # try:
                #     print(f"[KV] before tool: seq_len={before_len}, processed_tokens={processed_tokens}, gpu=({_gpu_mem_stats()})")
                # except Exception:
                #     pass
                py_call = coerce_python_call_message(last)
                tool_out = python_tool.process(py_call)
                tool_msgs = collect_tool_messages(tool_out)
                # 统计工具回传的内容长度（纯文本和 JSON 长度）
                try:
                    tool_chars = 0
                    for m in tool_msgs:
                        c = getattr(m, "content", "")
                        if isinstance(c, str):
                            tool_chars += len(c)
                    # print(f"[TOOL] python returned messages={len(tool_msgs)}, total_chars={tool_chars}")
                except Exception:
                    pass
                messages.extend(tool_msgs)
                # try:
                #     after_len = _kv_seq_len(past_key_values)
                # except Exception:
                #     after_len = -1
                # try:
                #     # 同时打印此刻渲染出来的前缀 token 长度，便于对照（不会触发生成）
                #     cur_ids_probe = encoding.render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)
                #     print(f"[KV] after tool: seq_len={after_len}, processed_tokens={processed_tokens}, prompt_tokens={len(cur_ids_probe)}, gpu=({_gpu_mem_stats()})")
                # except Exception:
                #     print(f"[KV] after tool: seq_len={after_len}, processed_tokens={processed_tokens}, gpu=({_gpu_mem_stats()})")
            elif base == "browser":
                # print(f"\n[TOOL DISPATCH] -> {name}")
                try:
                    before_len = _kv_seq_len(past_key_values)
                except Exception:
                    before_len = -1
                try:
                    print(f"[KV] before tool: seq_len={before_len}, processed_tokens={processed_tokens}, gpu=({_gpu_mem_stats()})")
                except Exception:
                    pass
                tool_out = browser_tool.process(last)
                tool_msgs = collect_tool_messages(tool_out)
                try:
                    tool_chars = 0
                    for m in tool_msgs:
                        c = getattr(m, "content", "")
                        if isinstance(c, str):
                            tool_chars += len(c)
                    print(f"[TOOL] browser returned messages={len(tool_msgs)}, total_chars={tool_chars}")
                except Exception:
                    pass
                messages.extend(tool_msgs)
                # try:
                #     after_len = _kv_seq_len(past_key_values)
                # except Exception:
                #     after_len = -1
                # try:
                #     cur_ids_probe = encoding.render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)
                #     print(f"[KV] after tool: seq_len={after_len}, processed_tokens={processed_tokens}, prompt_tokens={len(cur_ids_probe)}, gpu=({_gpu_mem_stats()})")
                # except Exception:
                #     print(f"[KV] after tool: seq_len={after_len}, processed_tokens={processed_tokens}, gpu=({_gpu_mem_stats()})")
            else:
                # 未知工具：回注错误消息，防止死循环
                messages.append(
                    Message.from_role_and_content(
                        Role.TOOL, json.dumps({"error": f"tool {name} not available"})
                    ).with_channel("commentary").with_recipient(name).with_content_type("json")
                )

            continue

        # 没有工具调用：检查是否已经给出 final；若未给出，则继续生成
        chan = (last.channel or "").strip().lower()
        if chan == "final":
            break
        else:
            # 既不是工具调用，也不是 final，继续下一轮生成
            continue
    else:
        raise RuntimeError("Exceeded MAX_STEPS without reaching <|return|>")


if __name__ == "__main__":
    main()
