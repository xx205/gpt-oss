# -*- coding: utf-8 -*-
# pip install -U jupyter_client ipykernel pyzmq
import asyncio
import json
from queue import Empty
from typing import AsyncGenerator, Optional

from jupyter_client import KernelManager
from openai_harmony import Message, Role


def coerce_python_call_message(last_msg: Message) -> Message:
    """把模型的工具调用消息规范化为“纯代码”的 python 调用消息。"""
    raw = last_msg.content[0].text if last_msg.content else ""
    try:
        obj = json.loads(raw)
        code = obj.get("code", raw)
    except json.JSONDecodeError:
        code = raw
    return (
        Message.from_role_and_content(Role.ASSISTANT, code)
        .with_channel("commentary")
        .with_recipient("python")
    )


class JupyterPythonTool:
    """用本机 Jupyter kernel 作为 GPT-OSS 的 python 工具后端（有状态）。

    - 单实例持有一个内核，会话内多次调用共享变量/导入/文件等状态
    - process(...) 返回 *异步生成器*，以便与原 Docker 版接口保持一致
    """

    def __init__(self, kernel_name: str = "python3", exec_timeout_s: float = 60.0):
        self.kernel_name = kernel_name
        self.exec_timeout_s = exec_timeout_s
        self.km: Optional[KernelManager] = None
        self.kc = None
        self._ensure_kernel()

    def _ensure_kernel(self) -> None:
        if self.km is not None:
            return
        self.km = KernelManager(kernel_name=self.kernel_name)
        self.km.start_kernel()
        # 使用阻塞客户端更简单稳定；我们把阻塞调用放在线程里或短轮询即可
        self.kc = self.km.blocking_client()
        self.kc.start_channels()

    def shutdown(self) -> None:
        try:
            if self.kc:
                self.kc.stop_channels()
        finally:
            if self.km:
                try:
                    self.km.shutdown_kernel(now=True)
                except Exception:
                    pass
            self.km = None
            self.kc = None

    def _run_code_blocking(self, code: str) -> str:
        """在阻塞客户端里执行一段代码，收集 stdout/结果/错误为纯文本。"""
        msg_id = self.kc.execute(code, allow_stdin=False, stop_on_error=False)
        stdout_chunks = []
        result_chunks = []
        error_chunks = []

        # 轮询 IOPub，直到看到 status: idle
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=0.2)
            except Empty:
                continue

            mtype = msg.get("msg_type")
            content = msg.get("content", {})

            if mtype == "stream":
                if content.get("name") == "stdout":
                    stdout_chunks.append(content.get("text", ""))
                elif content.get("name") == "stderr":
                    error_chunks.append(content.get("text", ""))

            elif mtype in ("execute_result", "display_data"):
                # 取 text/plain 作为可读结果
                data = content.get("data", {})
                text = data.get("text/plain") or data.get("text/markdown") or ""
                if text:
                    result_chunks.append(text)

            elif mtype == "error":
                tb = "\n".join(content.get("traceback") or [])
                if not tb:
                    tb = f"{content.get('ename','')}: {content.get('evalue','')}"
                error_chunks.append(tb)

            elif mtype == "status" and content.get("execution_state") == "idle":
                break

        out = ""
        if stdout_chunks:
            out += "".join(stdout_chunks)
        if result_chunks:
            # 与 Jupyter 协议一致：表达式值会以 execute_result/ display_data 形式出现
            # 这里直接附在后面，便于肉眼检查
            out += "".join(result_chunks)
        if error_chunks and not out:
            out = "".join(error_chunks)
        return out

    async def process(self, py_call_msg: Message) -> AsyncGenerator[Message, None]:
        """与 Docker 版保持相同签名：异步生成器，产出一条工具消息。"""
        code = py_call_msg.content[0].text if py_call_msg.content else ""
        # 把阻塞执行放到线程池，避免阻塞事件循环
        text = await asyncio.to_thread(self._run_code_blocking, code)
        yield (
            Message.from_role_and_content(Role.TOOL, text)
            .with_channel("commentary")
            .with_recipient("python")
        )


async def main() -> None:
    # 伪造一条“模型要调用 python 工具”的消息
    model_msg = (
        Message.from_role_and_content(Role.ASSISTANT, json.dumps({"code": "1 + 1"}))
        .with_channel("commentary")
        .with_recipient("python")
    )

    print(model_msg.to_json())

    python_tool = JupyterPythonTool(kernel_name="python3")
    tool_message = coerce_python_call_message(model_msg)

    try:
        async for m in python_tool.process(tool_message):
            # 显示 python 工具的输出
            print(m.content[0].text, end="")
    finally:
        python_tool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
