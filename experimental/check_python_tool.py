# -*- coding: utf-8 -*-
"""Minimal check for PythonTool functionality."""

import asyncio
import json

from gpt_oss.tools.python_docker.docker_tool import PythonTool
from openai_harmony import Message, Role


def coerce_python_call_message(last_msg: Message) -> Message:
    """Extract Python code from a model message and prepare it for PythonTool.

    The model may send JSON like {"code": "print(2+2)"}.  PythonTool expects the
    message's content to be plain python source code, so we parse the JSON and
    forward only the code string.
    """
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


async def main() -> None:
    """Simulate a tool call for ``print(1 + 1)`` and display its output."""
    model_msg = (
        Message.from_role_and_content(Role.ASSISTANT, json.dumps({"code": "print(1 + 1)"}))
        .with_channel("commentary")
        .with_recipient("python")
    )

    python_tool = PythonTool()
    tool_message = coerce_python_call_message(model_msg)

    async for m in python_tool.process(tool_message):
        if m.content:
            print(m.content[0].text, end="")


if __name__ == "__main__":
    asyncio.run(main())
