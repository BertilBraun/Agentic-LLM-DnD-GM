from fastapi import APIRouter, Request
from pydantic import BaseModel
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from ..embedder import embed
from ..qdrant_client import get_qdrant, COLLECTION
from shared.schemas import OkOut

router = APIRouter()

SNIPPET_LEN = 512


class StoreIn(BaseModel):
    turn_id: str
    text: str
    role: str


class RecallIn(BaseModel):
    query: str
    top_k: int = 8


class RecallOut(BaseModel):
    context: str


@router.post('/tools/store', response_model=OkOut)
async def store(body: StoreIn, request: Request) -> OkOut:
    campaign_id: str = request.state.campaign_id
    vector = await embed(body.text)
    client = get_qdrant()
    await client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=body.turn_id,
                vector=vector,
                payload={
                    'campaign_id': campaign_id,
                    'turn_id': body.turn_id,
                    'role': body.role,
                    'text_snippet': body.text[:SNIPPET_LEN],
                },
            )
        ],
    )
    return OkOut()


@router.post('/tools/recall', response_model=RecallOut)
async def recall(body: RecallIn, request: Request) -> RecallOut:
    campaign_id: str = request.state.campaign_id
    query_vector = await embed(body.query)
    client = get_qdrant()
    results = await client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=Filter(must=[FieldCondition(key='campaign_id', match=MatchValue(value=campaign_id))]),
        limit=body.top_k,
        with_payload=True,
    )
    lines = [f'[{r.payload["role"]}] {r.payload["text_snippet"]}' for r in results]
    return RecallOut(context='\n'.join(lines))
