# -*- coding: utf-8 -*-
"""Minimal check for SimpleBrowserTool functionality."""

import asyncio
import json

from gpt_oss.tools.simple_browser import SimpleBrowserTool, ExaBackend
from openai_harmony import Message, Role


async def main() -> None:
    """Simulate a search call and display the resulting text."""
    backend = ExaBackend(source="web")  # requires EXA_API_KEY
    browser = SimpleBrowserTool(backend=backend)

    model_msg = (
        Message.from_role_and_content(Role.ASSISTANT, json.dumps({"query": "OpenAI"}))
        .with_channel("commentary")
        .with_recipient("browser.search")
    )

    async for m in browser.process(model_msg):
        if m.content:
            print(m.content[0].text, end="")


if __name__ == "__main__":
    asyncio.run(main())
