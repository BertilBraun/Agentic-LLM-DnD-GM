import json
from ..redis_client import get_redis


async def publish_event(campaign_id: str, event: dict) -> None:
    redis = get_redis()
    await redis.publish(f"sse:campaign:{campaign_id}", json.dumps(event))
