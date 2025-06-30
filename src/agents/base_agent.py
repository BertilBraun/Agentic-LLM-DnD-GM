"""Common agent interface and shared utilities.

All concrete agents in the system (e.g. MasterAgent, SceneAgent) must inherit
from :class:`BaseAgent` to guarantee a uniform API for prompting and memory
management.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


class Memory(Protocol):
    """Lightweight protocol for memory objects stored by agents."""

    content: str
    metadata: Dict[str, Any]

    def __str__(self) -> str: ...  # noqa: D401


@dataclass
class Message:
    """Representation of a single conversational turn."""

    role: str  # "user" | "assistant" | "system"
    content: str


class BaseAgent(abc.ABC):
    """Abstract base class that defines the core agent interface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._memory: List[Memory] = []
        self._history: List[Message] = []

    # ------------------------------------------------------------------
    # Abstract API methods
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def prompt(self, user_input: str) -> str:
        """Generate a response given *user_input*."""

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def update_memory(self, memory: Memory) -> None:
        """Append *memory* to the agent's internal store."""
        self._memory.append(memory)

    def get_memory(self) -> List[Memory]:
        """Return the list of stored memories."""
        return list(self._memory)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the interaction history."""
        self._history.append(Message(role=role, content=content))

    def get_history(self) -> List[Message]:
        """Return conversation history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>" 