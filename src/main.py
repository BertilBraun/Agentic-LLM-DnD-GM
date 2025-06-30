"""Entry point for LLM-DnD application.

At startup this script searches for the most recent markdown save file
(`*.dnd-save.md`) in the `saves/` directory. If found, it restores a
`MasterAgent` from that file; otherwise it creates a fresh campaign state.

This minimal CLI demonstrates loading and saving but does not implement the
full gameplay loop yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from agents.master_agent import MasterAgent
from persistence.storage import load_campaign, save_campaign

SAVES_DIR = Path("saves")


def find_latest_save() -> Optional[Path]:
    """Return path to most recently modified save file, or None."""
    if not SAVES_DIR.exists():
        return None
    files = list(SAVES_DIR.glob("*.dnd-save.md"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-DnD Master Controller")
    parser.add_argument("--new", action="store_true", help="Start a new campaign (ignore saves)")
    args = parser.parse_args()

    master = MasterAgent()

    if not args.new:
        save_path = find_latest_save()
        if save_path:
            print(f"[+] Loading campaign from {save_path}")
            load_campaign(save_path, master)
        else:
            print("[+] No existing save found – starting new campaign")
    else:
        print("[+] New campaign requested – ignoring saves")

    # Demonstration: show loaded world state and plan
    print("World state:", master.world_state)
    print("Story plan:", master.story_plan)

    # Example of interacting with MasterAgent directly
    try:
        while True:
            user_in = input("You > ")
            if user_in.lower() in {"quit", "exit"}:
                break
            reply = master.prompt(user_in)
            print("GM >", reply)
    except KeyboardInterrupt:
        pass
    finally:
        # Save progress
        path = save_campaign(master)
        print(f"[+] Campaign saved to {path}")


if __name__ == "__main__":
    main() 