"""Voice-driven D&D framework using Gemini-2.5-Flash"""

from __future__ import annotations

import time
import asyncio
from typing import Type, TypeVar

from pydantic import BaseModel
from openai import OpenAI

from googletrans import Translator

from config import GEMINI_API_KEY, LANGUAGE, MODEL


# ───────────────────────────────────────────────────────────────
# LLM client
# ───────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)


def _translate_prompt(messages: list[dict[str, str]], language: str) -> list[dict[str, str]]:
    if language == 'en':
        return messages

    contents = [message['content'] for message in messages]

    async def translate():
        async with Translator() as translator:
            translations = await translator.translate(contents, dest=language)
            return [translation.text for translation in translations]

    translated_contents = asyncio.run(translate())

    return [
        {
            'role': message['role'],
            'content': translated_contents[i],
        }
        for i, message in enumerate(messages)
    ]


def llm_chat(messages: list[dict[str, str]]) -> str:
    messages = _translate_prompt(messages, LANGUAGE)

    messages = [
        {'role': 'system', 'content': 'You will always respond in the same language as the messages you are given.'}
    ] + messages

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
    messages = _translate_prompt(messages, LANGUAGE)

    messages = [
        {'role': 'system', 'content': 'You will always respond in the same language as the messages you are given.'}
    ] + messages

    start = time.time()
    res = client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,  # type: ignore
        response_format=response_format,
    )
    print(f'LLM parse took {time.time() - start:.2f}s')
    assert res.choices[0].message.parsed is not None
    return res.choices[0].message.parsed
