"""LangChain adapter utilities for LLM-driven agents.

Provides a helper to build a `langchain` `Chain` around any `BaseAgent`. The
chain handles conversation memory via `ConversationBufferMemory` and uses
LangChain's `ChatOpenAI` wrapper for model calls.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain.chat_models import ChatOpenAI  # type: ignore
from langchain.memory import ConversationBufferMemory  # type: ignore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate  # type: ignore
from langchain.schema import AIMessage, HumanMessage  # type: ignore
from langchain.chains import LLMChain  # type: ignore

from .base_agent import BaseAgent

__all__ = ["build_agent_chain"]


def build_agent_chain(
    agent: BaseAgent,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.7,
    model_name: str = "gpt-3.5-turbo",
    verbose: bool = False,
) -> LLMChain:
    """Return an LLMChain that routes user input via *agent* conversation memory.

    The chain internally stores history in LangChain `ConversationBufferMemory`
    while still mirroring the messages to the agent's history store.
    """

    if system_instruction is None:
        system_instruction = (
            "You are a helpful assistant participating in a D&D campaign. "
            "Follow the narrative guidance from the Dungeon Master."
        )

    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_instruction),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Callback to mirror messages back to the agent store
    def _mirror_memory(_input: Dict[str, Any], _output: str) -> None:  # type: ignore[override]
        agent.add_message("user", _input["input"])
        agent.add_message("assistant", _output)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        callbacks=[_mirror_memory],  # type: ignore[arg-type]
        verbose=verbose,
    )

    return chain 