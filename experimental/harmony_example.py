# -*- coding: utf-8 -*-
# pip install -U torch transformers openai-harmony gpt-oss
import asyncio
import inspect
import json
import sys
import threading
import gc
from collections.abc import Iterable
from datetime import date

import torch
from gpt_oss.tools.simple_browser import SimpleBrowserTool, ExaBackend
from gpt_oss.tools.python_docker.docker_tool import PythonTool
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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

def setup_runtime(_model_id: str | None = None) -> None:
    """惰性初始化分词器、模型、编码与工具（Browser/Python）。"""
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
    device_map = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype="auto",
    )
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
    python_tool = PythonTool()

    # 写回全局
    globals()["browser_tool"] = browser_tool
    globals()["python_tool"] = python_tool

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
    .with_python_tool()  # 声明可用 Python 工具命名空间
    .with_browser_tool() # 声明可用 Browser 工具命名空间
)

developer_msg = (
    DeveloperContent.new().with_instructions(
        "始终在一次 Python 工具调用中完成计算并使用 print 输出结果"
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
    max_new_tokens=8192,
    past_key_values=None,
    processed_tokens=0,
):
    """Generate one step using cached KV state.

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

    convo = Conversation.from_messages(msgs)
    input_token_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    new_input_ids = input_token_ids[processed_tokens:]

    # Move inputs to the model's device before generation
    input_ids = torch.tensor([new_input_ids], dtype=torch.long).to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        past_key_values=past_key_values,
        return_dict_in_generate=True,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True,
    )
    eids = _collect_eos_ids()
    if eids:
        gen_kwargs["eos_token_id"] = eids

    with torch.inference_mode():
        if stream:
            # >>> 回滚这里：打印“原始输出”，不要隐藏特殊 token <<<
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,  # ← 不隐藏特殊 token
                decode_kwargs=dict(clean_up_tokenization_spaces=False),
            )
            gen_kwargs["streamer"] = streamer
            out_ids = []
            box = {}

            def _runner():
                output = model.generate(**gen_kwargs)
                box["out"] = output
                seqs = output.sequences[0]
                out_ids.extend(seqs.tolist()[input_ids.shape[-1]:])

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            for chunk in streamer:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            t.join()
            output = box.get("out", None)
        else:
            output = model.generate(**gen_kwargs)
            seqs = output.sequences[0]
            out_ids = seqs.tolist()[input_ids.shape[-1]:]

    # 取到最小必要集合后，尽早丢掉临时对象引用
    past_key_values = output.past_key_values if output is not None else None
    try:
        del output
    except Exception:
        pass
    try:
        del seqs
    except Exception:
        pass
    try:
        del streamer
    except Exception:
        pass
    try:
        box.clear()
    except Exception:
        pass

    # 同步 + 清理缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _release_cuda()

    processed_tokens = len(input_token_ids) + len(out_ids)
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
        except Exception:
            pass

        # 找到底层缓存并小拷贝为 list[(k,v)]
        for attr in ("caches", "layers", "kv_cache", "key_value_caches"):
            if hasattr(pkv, attr):
                caches = getattr(pkv, attr)
                new_list = []
                for layer in caches:
                    k = getattr(layer, "key_cache", None)
                    v = getattr(layer, "value_cache", None)
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        new_k = k[..., :L, :].contiguous().clone()
                        new_v = v[..., :L, :].contiguous().clone()
                        new_list.append((new_k, new_v))
                    else:
                        return None
                return new_list
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

# =========================
# 5) 工具循环
# =========================
MAX_STEPS = 32

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
                    # 复用到 lcp，并对 KV 做“物理紧凑化”以释放旧大缓存
                    old_pkv = past_key_values
                    truncated = _truncate_past_key_values(old_pkv, lcp)
                    if truncated is None:
                        past_key_values = None
                        processed_tokens = 0
                        last_seen_ids = None
                    else:
                        past_key_values = truncated
                        processed_tokens = lcp
                        last_seen_ids = cur_prefix_ids[:lcp]
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

        last = gen_msgs[-1]
        name = (last.recipient or "").strip()

        if name:
            # 兼容 "python", "python.exec", "python code" 等变体
            base = name.split(None, 1)[0].split(".", 1)[0]
            if base == "python":
                print(f"\n[TOOL DISPATCH] -> {name}")
                py_call = coerce_python_call_message(last)
                tool_out = python_tool.process(py_call)
                tool_msgs = collect_tool_messages(tool_out)
                messages.extend(tool_msgs)
            elif base == "browser":
                print(f"\n[TOOL DISPATCH] -> {name}")
                tool_out = browser_tool.process(last)
                tool_msgs = collect_tool_messages(tool_out)
                messages.extend(tool_msgs)
            else:
                # 未知工具：回注错误消息，防止死循环
                messages.append(
                    Message.from_role_and_content(
                        Role.TOOL, json.dumps({"error": f"tool {name} not available"})
                    ).with_channel("commentary").with_recipient(name).with_content_type("json")
                )

            continue

        # 没有工具调用，通常代表模型已给出 final（或下一轮继续）
        break
    else:
        raise RuntimeError("Exceeded MAX_STEPS without reaching <|return|>")


if __name__ == "__main__":
    main()
