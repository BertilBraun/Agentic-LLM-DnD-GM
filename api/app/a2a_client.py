"""Thin A2A JSON-RPC client — sends tasks/send to agent services."""
import uuid
import httpx


async def send_task(agent_url: str, campaign_id: str, message: str, timeout: float = 120) -> dict:
    task_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {"task_id": task_id, "campaign_id": campaign_id, "message": message},
        "id": 1,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(agent_url + "/", json=payload)
        resp.raise_for_status()
        return resp.json()
