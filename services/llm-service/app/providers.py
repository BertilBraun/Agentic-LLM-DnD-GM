"""LLM provider adapters — Gemini, OpenAI, Anthropic."""

from __future__ import annotations

import abc
import json
import os
from typing import Any


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def generate(
        self,
        messages: list[dict],
        response_format: str,
        response_json_schema: dict | None = None,
    ) -> tuple[str, int, int]:
        """Returns (text, tokens_in, tokens_out)."""


# ─── Gemini ──────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        from google import genai

        self.client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        self.model = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-lite-preview')

    async def generate(
        self,
        messages: list[dict],
        response_format: str,
        response_json_schema: dict | None = None,
    ) -> tuple[str, int, int]:
        from google.genai import types

        system_parts = [m['content'] for m in messages if m['role'] == 'system']
        contents = _gemini_contents(messages)

        config_kwargs: dict[str, Any] = {
            'system_instruction': '\n\n'.join(system_parts) if system_parts else None,
        }
        if response_json_schema:
            config_kwargs['response_mime_type'] = 'application/json'
            config_kwargs['response_json_schema'] = response_json_schema
        elif response_format == 'json':
            config_kwargs['response_mime_type'] = 'application/json'
        else:
            config_kwargs['response_mime_type'] = 'text/plain'

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text or ''
        usage = response.usage_metadata
        return text, (usage.prompt_token_count or 0), (usage.candidates_token_count or 0)


def _gemini_contents(messages: list[dict]) -> list[Any]:
    from google.genai import types

    result = []
    for m in messages:
        if m['role'] == 'system':
            continue  # handled via system_instruction in config
        elif m['role'] == 'user':
            result.append(types.Content(role='user', parts=[types.Part(text=m['content'])]))
        elif m['role'] == 'assistant':
            result.append(types.Content(role='model', parts=[types.Part(text=m['content'])]))
    return result


# ─── OpenAI ──────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.model = os.environ.get('OPENAI_MODEL', 'gpt-4o')

    async def generate(
        self,
        messages: list[dict],
        response_format: str,
        response_json_schema: dict | None = None,
    ) -> tuple[str, int, int]:
        kwargs: dict = {}
        if response_json_schema:
            # Structured outputs with schema — strict=False avoids $defs/$ref constraints
            kwargs['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'response',
                    'schema': response_json_schema,
                    'strict': False,
                },
            }
        elif response_format == 'json':
            kwargs['response_format'] = {'type': 'json_object'}

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        text = response.choices[0].message.content or ''
        usage = response.usage
        return text, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)


# ─── Anthropic ───────────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        self.model = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6')

    async def generate(
        self,
        messages: list[dict],
        response_format: str,
        response_json_schema: dict | None = None,
    ) -> tuple[str, int, int]:
        system_parts = [m['content'] for m in messages if m['role'] == 'system']
        user_messages = [dict(m) for m in messages if m['role'] != 'system']
        system = '\n\n'.join(system_parts) if system_parts else ''

        # Anthropic has no native schema enforcement — inject into system prompt
        if response_json_schema:
            schema_instruction = 'Respond with valid JSON matching this exact schema:\n' + json.dumps(
                response_json_schema, indent=2
            )
            system = (system + '\n\n' + schema_instruction).strip()
        elif response_format == 'json':
            system = (system + '\n\nRespond with valid JSON only.').strip()

        kwargs: dict = {}
        if system:
            kwargs['system'] = system

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=8096,
            messages=user_messages,
            **kwargs,
        )
        text = response.content[0].text if response.content else ''
        usage = response.usage
        return text, (usage.input_tokens if usage else 0), (usage.output_tokens if usage else 0)


# ─── Factory ─────────────────────────────────────────────────────

_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        name = os.environ.get('LLM_PROVIDER', 'gemini')
        if name == 'openai':
            _provider = OpenAIProvider()
        elif name == 'anthropic':
            _provider = AnthropicProvider()
        else:
            _provider = GeminiProvider()
    return _provider
