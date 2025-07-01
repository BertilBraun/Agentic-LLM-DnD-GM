"""Voice-driven D&D framework using Gemini-2.5-Flash"""

from __future__ import annotations

import time
from typing import Type, TypeVar

from pydantic import BaseModel
from openai import OpenAI


from config import GEMINI_API_KEY, MODEL

# ───────────────────────────────────────────────────────────────
# 1 ·  LLM client
# ───────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)


def llm_chat(messages: list[dict[str, str]]) -> str:
    start = time.time()
    res = client.chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore
    )
    print(f'LLM call took {time.time() - start:.2f}s')
    assert res.choices[0].message.content is not None
    return res.choices[0].message.content


T = TypeVar('T', bound=BaseModel)


def llm_parse(messages: list[dict[str, str]], response_format: Type[T]) -> T:
    start = time.time()
    res = (
        client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,  # type: ignore
            response_format=response_format,
        )
        .choices[0]
        .message.parsed
    )
    print(f'LLM parse took {time.time() - start:.2f}s')
    assert res is not None
    return res
