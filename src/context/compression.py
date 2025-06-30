"""Utilities for summarising long conversation histories into compact memory strings.

These helpers perform token- (rough) based chunking of message history and call
an LLM (currently OpenAI ChatCompletion) to build hierarchical summaries. The
resulting summaries can be stored by agents to free up context while retaining
important information.
"""
from __future__ import annotations

import math
import os
from typing import Any, List, Sequence

import openai  # type: ignore

# ---------------------------------------------------------------------------
# Rough token estimation (fallback when tiktoken unavailable)
# ---------------------------------------------------------------------------
try:
    import tiktoken  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tiktoken = None  # type: ignore


def approx_tokens(text: str) -> int:
    """Very rough token estimate (≈ 4 chars/token)."""
    if tiktoken is not None:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def chunk_messages(messages: Sequence[dict[str, str]], max_tokens: int = 1500) -> list[list[dict[str, str]]]:  # type: ignore[valid-type]
    """Split *messages* into chunks whose token estimate ≤ *max_tokens*."""
    chunks: List[List[dict[str, str]]] = [[]]
    current_tokens = 0
    for msg in messages:
        msg_tokens = approx_tokens(msg["content"])
        if current_tokens + msg_tokens > max_tokens and chunks[-1]:
            # start new chunk
            chunks.append([])
            current_tokens = 0
        chunks[-1].append(msg)
        current_tokens += msg_tokens
    return chunks


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarize_chunk(chunk: Sequence[dict[str, str]], *, model: str = "gpt-3.5-turbo", target_tokens: int = 256) -> str:
    """Return summary string of a chunk of messages."""
    prompt = (
        "Summarise the following D&D conversation chunk into concise third-person prose. "
        "Include important plot details, character actions, and any new facts introduced."
    )
    messages = [{"role": "system", "content": prompt}] + list(chunk)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=target_tokens,
    )
    return resp.choices[0].message.content.strip()


def hierarchical_summary(messages: Sequence[dict[str, str]], chunk_tokens: int = 1500, *, model: str = "gpt-3.5-turbo") -> str:
    """Recursively summarise *messages* until summary fits under *chunk_tokens*."""
    # First pass: summarise chunks
    chunks = chunk_messages(messages, max_tokens=chunk_tokens)
    summaries = [summarize_chunk(chunk, model=model) for chunk in chunks]

    if len(summaries) == 1:
        return summaries[0]

    # Second pass: summarise combined summaries (should be short)
    merged_msgs = [{"role": "system", "content": s} for s in summaries]
    return hierarchical_summary(merged_msgs, chunk_tokens=chunk_tokens, model=model)


# ---------------------------------------------------------------------------
# Convenience for agent history → memory
# ---------------------------------------------------------------------------

def compress_history_to_memory(history: Sequence[Any], *, model: str = "gpt-3.5-turbo") -> str:
    """Generate single summary string memory from agent *history* (list of Message)."""
    msg_dicts = [
        {"role": m.role if hasattr(m, "role") else "user", "content": m.content if hasattr(m, "content") else str(m)}
        for m in history
    ]
    return hierarchical_summary(msg_dicts, model=model) 