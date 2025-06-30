"""Image generation wrapper for Runware Flux[schnell] synchronous API.

This module exposes a simple `generate_image()` function that takes a textual
prompt and returns a base64-encoded PNG (string). The underlying HTTP request
is synchronous; callers may run it in an executor if non-blocking behaviour is
needed.

Note: The real Runware client library/API details are hypothetical in this
context. This implementation uses `requests` to demonstrate the expected flow
and error handling.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass

import requests  # type: ignore

RUNWARE_ENDPOINT = os.getenv("RUNWARE_ENDPOINT", "https://api.runware.ai/generate")
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY", "")

DEFAULT_MODEL = os.getenv("RUNWARE_MODEL", "fluxschnell-v2")
DEFAULT_CFG_SCALE = float(os.getenv("RUNWARE_CFG", "7.0"))
DEFAULT_STEPS = int(os.getenv("RUNWARE_STEPS", "30"))
DEFAULT_RESOLUTION = os.getenv("RUNWARE_RES", "512x512")  # WIDTHxHEIGHT

__all__ = [
    "generate_image",
]


@dataclass
class ImageConfig:
    model: str = DEFAULT_MODEL
    steps: int = DEFAULT_STEPS
    cfg_scale: float = DEFAULT_CFG_SCALE
    resolution: str = DEFAULT_RESOLUTION


# Mutable global config instance used as fallback when parameters omitted.
_default_config = ImageConfig()


def configure(**kwargs) -> None:
    """Set global default image parameters.

    Example
    -------
    >>> configure(model="fluxschnell-v3", resolution="768x768", steps=40)
    """
    global _default_config
    for key, value in kwargs.items():
        if hasattr(_default_config, key):
            setattr(_default_config, key, value)
        else:
            raise AttributeError(f"Unknown ImageConfig field '{key}'")


def _build_payload(prompt: str, *, model: str, steps: int, cfg_scale: float, resolution: str) -> Dict[str, Any]:
    width, height = map(int, resolution.lower().split("x"))
    return {
        "model": model,
        "prompt": prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
    }


def generate_image(
    prompt: str,
    *,
    model: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    resolution: str | None = None,
    timeout: int = 60,
) -> str:
    """Generate an image and return it as a base64-encoded string.

    Raises RuntimeError on failure.
    """
    headers = {
        "Authorization": f"Bearer {RUNWARE_API_KEY}",
        "Content-Type": "application/json",
    }
    # Fallback to default config values if not provided
    cfg = _default_config
    payload = _build_payload(
        prompt,
        model=model or cfg.model,
        steps=steps or cfg.steps,
        cfg_scale=cfg_scale or cfg.cfg_scale,
        resolution=resolution or cfg.resolution,
    )

    try:
        resp = requests.post(RUNWARE_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:  # pragma: no cover
        raise RuntimeError(f"Network error contacting Runware API: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Runware API returned {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
        return data["image_base64"]  # type: ignore[index]
    except (ValueError, KeyError) as exc:
        raise RuntimeError("Invalid response format from Runware API") from exc


if __name__ == "__main__":
    test_prompt = "Epic fantasy landscape with towering mountains and a dragon in the sky, digital art"
    b64 = generate_image(test_prompt)
    print(b64[:100] + "... (truncated)") 