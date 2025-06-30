"""Prompt builder for image generation.

This helper ensures a consistent illustrative style across all generated images
and appends camera, lighting, and quality tags recommended for the Runware
Flux[schnell] model suite.
"""
from __future__ import annotations

from typing import List

__all__ = ["build_image_prompt"]

# Global style template (can be adjusted centrally)
BASE_STYLE = (
    "Illustration, high detail, soft cinematic lighting, concept art, "
    "vibrant colors, trending on artstation, 4k,"
)

NEGATIVE_PROMPT = (
    "(low quality, worst quality, photorealistic, disfigured, cropped, text, watermark)"
)


def build_image_prompt(scene_description: str, *, extra_tags: List[str] | None = None) -> str:
    """Return a prompt string combining scene description with style.

    Parameters
    ----------
    scene_description
        Free-form textual description of the scene or subject.
    extra_tags
        Optional list of additional style/concept tags appended after BASE_STYLE.
    """
    prompt_parts = [scene_description.rstrip(" ,.")]
    prompt_parts.append(BASE_STYLE)
    if extra_tags:
        prompt_parts.extend(extra_tags)
    prompt_parts.append("--no " + NEGATIVE_PROMPT)
    return ", ".join(prompt_parts)


if __name__ == "__main__":
    example = build_image_prompt("Ancient forest with glowing runes on stone pillars")
    print(example) 