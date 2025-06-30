"""SceneAgent – handles localized interaction within the broader campaign.

Each SceneAgent is instantiated per scene/encounter and references the shared
`MasterAgent` to ensure consistency with the global world state and story plan.
"""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime as _dt

import openai  # type: ignore

from .base_agent import BaseAgent, Message
from .master_agent import MasterAgent

__all__ = ["SceneAgent", "create_scene_agent"]


class SceneAgent(BaseAgent):
    """Agent focused on a single scene while accessing the MasterAgent."""

    def __init__(
        self,
        master: MasterAgent,
        scene_name: str,
        background: str,
        *,
        llm_model: str | None = None,
    ) -> None:
        super().__init__(name=scene_name)
        self.master = master
        self.background = background  # Brief setting and goals for the scene
        self._llm_model = llm_model or "gpt-3.5-turbo"

    # ------------------------------------------------------------------
    # Core prompt method
    # ------------------------------------------------------------------
    def prompt(self, user_input: str) -> str:  # noqa: D401
        """Generate a scene-specific response informed by MasterAgent state."""
        self.add_message("user", user_input)

        system_prompt = (
            f"You are roleplaying a D&D scene: {self.name}. "
            "Stay consistent with the global world state and story plan provided by the Dungeon Master."
        )

        # Derive context from master
        world_ctx = json_truncate(self.master.world_state)
        plan_ctx = " -> ".join(self.master.story_plan[:10])

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Scene background: {self.background}"},
            {"role": "system", "content": f"World state: {world_ctx}"},
            {"role": "system", "content": f"Story plan: {plan_ctx}"},
        ] + [vars(m) for m in self._history]

        try:
            response = openai.ChatCompletion.create(
                model=self._llm_model,
                messages=messages,
                temperature=0.85,
            )
            content = response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover
            content = f"[LLM error: {exc}; returning echo] {user_input}"

        self.add_message("assistant", content)
        return content

    # ------------------------------------------------------------------
    # Scene completion hook
    # ------------------------------------------------------------------
    def conclude(self, summary: str | None = None) -> None:
        """Optional cleanup: write scene memories back to MasterAgent."""
        if summary is None:
            summary = f"Scene '{self.name}' concluded."
        self.master.update_memory(
            type("_Mem", (), {"content": summary, "metadata": {"scene": self.name}})()  # type: ignore[arg-type]
        )

        # Auto-save campaign after scene conclusion
        if getattr(self.master, "autosave", False):
            from ..persistence.storage import save_campaign  # local import to avoid cycles

            save_campaign(
                self.master,
                scene_history=[{
                    "title": self.name,
                    "summary": summary,
                    "date": _iso_today(),
                }],
            )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_scene_agent(master_agent: MasterAgent, scene_name: str, background: str) -> SceneAgent:
    """Factory function to create a SceneAgent tied to *master_agent*."""
    return SceneAgent(master_agent, scene_name, background)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def json_truncate(obj: Any, limit: int = 400) -> str:  # noqa: ANN401 – generic helper
    """Quick-and-dirty json serialization truncated to *limit* chars."""
    import json  # local import to avoid top-level cost

    try:
        s = json.dumps(obj, ensure_ascii=False)[:limit]
    except TypeError:
        s = str(obj)[:limit]
    return s

def _iso_today() -> str:
    return _dt.utcnow().date().isoformat() 