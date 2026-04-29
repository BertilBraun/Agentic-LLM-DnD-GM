import os
from google import genai
from google.genai import types

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


async def embed(text: str) -> list[float]:
    client = get_client()
    response = await client.aio.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=1536),
    )
    return response.embeddings[0].values
