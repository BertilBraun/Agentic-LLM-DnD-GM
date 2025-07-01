"""Voice-driven D&D framework using Gemini-2.5-Flash"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import time
from runware import Runware, IImageInference

from config import RUNWARE_API_KEY

# Image generation setup
image_client = Runware(api_key=RUNWARE_API_KEY)
asyncio.run(image_client.connect())


NEGATIVE_PROMPT = 'blurry, low quality, distorted, multiple people, crowd, background characters, text, watermark'


def generate_image(prompt: str, negative_prompt: str = NEGATIVE_PROMPT, n_images: int = 1) -> list[Path]:
    """Generate images using Runware.
    Args:
        prompt: The prompt to generate the image from.
        negative_prompt: The negative prompt to prevent the image from generating.
        n_images: The number of images to generate.
    Returns:
        A list of filepaths to the generated images.
    """

    if len(prompt) > 3000:
        print('WARNING: Prompt is too long for image generation:', prompt)
        prompt = prompt[:3000]
    if len(negative_prompt) > 3000:
        print('WARNING: Negative prompt is too long for image generation:', negative_prompt)
        negative_prompt = negative_prompt[:3000]

    async def _generate_image():
        request = IImageInference(
            positivePrompt=prompt,
            negativePrompt=negative_prompt,
            model='runware:100@1',
            width=1024,
            height=1024,
            includeCost=True,
            steps=6,
            CFGScale=1,
            scheduler='FlowMatchEulerDiscreteScheduler',
            numberResults=n_images,
            outputType='base64Data',
            outputFormat='JPG',
        )

        images = await image_client.imageInference(requestImage=request)

        assert images is not None
        assert len(images) == n_images
        print(f'Generating {len(images)} images costed {sum(image.cost or 0.0 for image in images)}')

        file_paths: list[Path] = []
        for i, image in enumerate(images):
            if image.imageBase64Data is None:
                print(f'WARNING: Image {i} is None')
                continue

            filename = f'{int(time.time())}_{i}.png'
            filename_path = Path('cache/images') / filename
            filename_path.parent.mkdir(parents=True, exist_ok=True)

            with open(filename_path, 'wb') as f:
                f.write(base64.b64decode(image.imageBase64Data))

            file_paths.append(filename_path)

        return file_paths

    return asyncio.run(_generate_image())
