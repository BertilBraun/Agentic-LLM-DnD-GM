"""Persistence helpers for saving and loading campaign state.

This module follows the schema described in `docs/campaign_save_schema.md`.
It produces human-readable Markdown files (`*.dnd-save.md`) and can restore
`MasterAgent` state on resume.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import List

import yaml  # type: ignore

from ..agents.master_agent import MasterAgent

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")


def save_campaign(
    master: MasterAgent,
    *,
    scene_history: List[dict[str, str]] | None = None,
    open_threads: List[str] | None = None,
    out_dir: Path | str = "saves",
) -> Path:
    """Serialize *master* state and additional sections to a Markdown file.

    Returns
    -------
    Path
        Path to the written save file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": 1,
        "campaign": master.world_state.get("campaign_name", master.name),
        "created": master.world_state.get("created", _iso_now()),
        "last_played": _iso_now(),
    }

    campaign_slug = _slug(meta["campaign"])
    filename = f"{campaign_slug}_{meta['last_played'].replace(':', '-')}.dnd-save.md"
    path = out_dir / filename

    md_lines: list[str] = []

    # Metadata (YAML)
    md_lines.append("# Metadata")
    md_lines.append("---")
    md_lines.extend(yaml.safe_dump(meta, sort_keys=False).splitlines())
    md_lines.append("---\n")

    # World State
    md_lines.append("# World State")
    md_lines.append("---")
    world_state_str = yaml.safe_dump(master.world_state, sort_keys=False)
    md_lines.extend(world_state_str.splitlines())
    md_lines.append("---\n")

    # Story Plan
    md_lines.append("# Story Plan")
    md_lines.append("---")
    for i, beat in enumerate(master.story_plan, 1):
        md_lines.append(f"{i}. {beat}")
    md_lines.append("---\n")

    # Scene History
    if scene_history:
        md_lines.append("# Scene History")
        md_lines.append("---")
        for scene in scene_history:
            summary = scene.get("summary", "")
            title = scene.get("title", "Unnamed Scene")
            date = scene.get("date", _iso_now().split("T")[0])
            link = scene.get("transcript", "")
            md_lines.append("<details>")
            md_lines.append(f"<summary>{date} – \"{title}\"</summary>\n")
            md_lines.append(f"**Summary**: {summary}\n")
            if link:
                md_lines.append(f"**Transcript**: [[{link}]]\n")
            md_lines.append("</details>\n")
        md_lines.append("---\n")

    # Open Threads
    if open_threads:
        md_lines.append("# Open Threads")
        md_lines.append("---")
        for thread in open_threads:
            md_lines.append(f"- {thread}")
        md_lines.append("---\n")

    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

_SECTION_PATTERN = re.compile(r"^# (.+)$", re.MULTILINE)


def load_campaign(path: Path | str, master: MasterAgent) -> None:
    """Populate *master* with state from a save file at *path*."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Split into sections
    sections = {}
    headings = list(_SECTION_PATTERN.finditer(text))
    for idx, match in enumerate(headings):
        start = match.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        sections[match.group(1).strip()] = text[start:end].strip()

    # Metadata (not used yet)

    # World State YAML
    ws_block = sections.get("World State", "")
    if ws_block:
        ws_block = ws_block.strip("- \n")
        master.world_state = yaml.safe_load(ws_block) or {}

    # Story Plan numbered list
    sp_block = sections.get("Story Plan", "")
    story_plan = []
    for line in sp_block.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.\s", line):
            story_plan.append(line.split(" ", 1)[1])
    master.story_plan = story_plan

    # TODO: Scene History & Open Threads parsing as needed


__all__ = [
    "save_campaign",
    "load_campaign",
] 