import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

COLLECTION = "campaign_turns"
VECTOR_SIZE = 1536

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://qdrant:6333"))
    return _client


async def ensure_collection() -> None:
    client = get_qdrant()
    existing = {c.name for c in (await client.get_collections()).collections}
    if COLLECTION not in existing:
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        await client.create_payload_index(
            collection_name=COLLECTION,
            field_name="campaign_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
