"""Real-time log streaming CLI (tail -f style).

Usage:
    python -m src.ui.logger_cli --file path/to/logfile.log

If no --file provided, defaults to `campaign.log`. The CLI continuously prints
new log lines with optional ANSI coloring powered by `colorama` (auto-fallback
if not available).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Optional color support
try:
    from colorama import Fore, Style, init as color_init  # type: ignore

    color_init()

    def _format(line: str) -> str:
        if "ERROR" in line:
            return f"{Fore.RED}{line}{Style.RESET_ALL}"
        if "WARNING" in line:
            return f"{Fore.YELLOW}{line}{Style.RESET_ALL}"
        if "INFO" in line:
            return f"{Fore.GREEN}{line}{Style.RESET_ALL}"
        return line

except ModuleNotFoundError:  # pragma: no cover

    def _format(line: str) -> str:  # type: ignore[override]
        return line


def tail_f(path: Path, *, sleep_sec: float = 0.4) -> None:
    """Continuously yield new lines appended to *path* (like `tail -f`)."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        # Seek to end
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(sleep_sec)
                continue
            print(_format(line.rstrip("\n")))
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live log viewer")
    parser.add_argument("--file", default="campaign.log", help="Log file to follow")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[logger_cli] File {path} does not exist. Waiting for it to be created…")
        while not path.exists():
            time.sleep(0.5)
    print(f"[logger_cli] Following {path} (Ctrl+C to stop)")
    try:
        tail_f(path)
    except KeyboardInterrupt:
        print("\n[logger_cli] Stopped.")


if __name__ == "__main__":
    main() 