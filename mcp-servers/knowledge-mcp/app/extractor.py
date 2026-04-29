import json
import os
import httpx

LLM_SERVICE_URL = os.environ.get('LLM_SERVICE_URL', 'http://llm-service:9001')

EXTRACTION_SYSTEM = """Extract named entities and relationships from the D&D narrative text.
Return JSON matching this exact schema:
{
  "nodes": [
    {"label": "NPC|Location|Faction|Item|Event", "name": "string", "properties": {...}}
  ],
  "relationships": [
    {"from_label": "...", "from_name": "...", "type": "LIVES_IN|MEMBER_OF|ALLIED_WITH|HOSTILE_TO|CONTROLS|VISITED|INVOLVES|OWNS|LOCATED_IN", "to_label": "...", "to_name": "...", "properties": {...}}
  ]
}
Only extract clearly named entities. Return empty lists if none found."""


async def extract_entities(text: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f'{LLM_SERVICE_URL}/generate',
            json={
                'messages': [
                    {'role': 'system', 'content': EXTRACTION_SYSTEM},
                    {'role': 'user', 'content': text},
                ],
                'response_format': 'json',
                'cache': True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    try:
        return json.loads(data['text'])
    except (json.JSONDecodeError, KeyError):
        return {'nodes': [], 'relationships': []}
