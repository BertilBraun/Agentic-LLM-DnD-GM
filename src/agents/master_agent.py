"""MasterAgent – central controller maintaining campaign world state and story plan.

The MasterAgent is persistent across an entire play-through. It keeps a high-level
record of the world, NPCs, quests, and overarching narrative beats. SceneAgents
spawned for individual encounters reference this shared state but focus on local
context.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import openai  # type: ignore

from .base_agent import BaseAgent, Memory, Message

__all__ = ["MasterAgent"]


class MasterAgent(BaseAgent):
    """Persistent agent that tracks global campaign information."""

    SAVE_FILE = Path("campaign_state.json")

    def __init__(
        self,
        name: str = "GM",
        *,
        llm_model: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        super().__init__(name)
        self.world_state: Dict[str, Any] = {}
        self.story_plan: List[str] = []  # Narrative outline – list of beats
        self._llm_model = llm_model or "gpt-3.5-turbo"
        if openai_api_key:
            openai.api_key = openai_api_key

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> None:
        """Serialise world state and story plan to *path* (default SAVE_FILE)."""
        path = path or self.SAVE_FILE
        state = {
            "world_state": self.world_state,
            "story_plan": self.story_plan,
            "memory": [m.content for m in self._memory],
            "history": [vars(msg) for msg in self._history],
        }
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False))

    def load(self, path: Path | None = None) -> None:
        """Load state from disk and populate agent attributes."""
        path = path or self.SAVE_FILE
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.world_state = data.get("world_state", {})
        self.story_plan = data.get("story_plan", [])
        # memory/history reload simplified – only content
        self._memory = [Memory(content=c, metadata={}) for c in data.get("memory", [])]  # type: ignore[arg-type]
        self._history = [Message(**m) for m in data.get("history", [])]

    # ------------------------------------------------------------------
    # World-state helpers
    # ------------------------------------------------------------------
    def update_world_state(self, key: str, value: Any) -> None:
        self.world_state[key] = value

    def get_world_state(self) -> Dict[str, Any]:
        return dict(self.world_state)

    # ------------------------------------------------------------------
    # LLM prompting
    # ------------------------------------------------------------------
    def prompt(self, user_input: str) -> str:  # noqa: D401 – matches base interface
        """Generate narrative response using the configured LLM."""
        self.add_message("user", user_input)

        system_prompt = (
            "You are the Dungeon Master for a D&D campaign. "
            "Maintain narrative consistency with the provided world state and story plan."
        )
        messages = [
            {"role": "system", "content": system_prompt},
        ] + [vars(m) for m in self._history]

        try:
            response = openai.ChatCompletion.create(
                model=self._llm_model,
                messages=messages,
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover – network error fallback
            content = f"[LLM error: {exc}; returning echo] {user_input}"

        # Store and return
        self.add_message("assistant", content)
        return content 