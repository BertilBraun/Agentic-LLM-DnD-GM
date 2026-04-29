"""Memory agent — mirrors SimpleMemorySystem from src/memory.py (Section 5.6)."""
from __future__ import annotations

import logging
import os
import uuid

from shared.helpers import call_mcp, call_llm_json

STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
MEMORY_MCP_URL = os.environ.get("MEMORY_MCP_URL", "http://memory-mcp:8002")

STM_THRESHOLD = 5

logger = logging.getLogger(__name__)

CUTOFF_SYSTEM = """Analyze the following recent D&D session events and determine if this is a good cutoff point for compressing memory.

Consider these factors:
- Have the players moved to a significantly different location/context?
- Has a major scene or encounter concluded?
- Are they transitioning between story beats (rest, travel, new chapter)?
- Have they left NPCs/locations they likely won't return to soon?
- Is there a natural narrative break?

A good cutoff point means the recent events form a cohesive "session" that can be summarized without losing important context.

Respond with JSON: {"reason": "...", "should_compress": true|false}"""

COMPRESSION_SYSTEM = """You are helping manage memory for a D&D campaign. Compress the short-term memory into the long-term memory.

Your task:
1. Update the long-term memory by integrating the short-term events
2. Preserve all important information: decisions, discoveries, relationships, quests, consequences
3. Use clear markdown formatting with sections:
   - ## Key Decisions & Actions
   - ## Characters Met
   - ## Locations Visited
   - ## Quests & Objectives
   - ## Important Information Learned
   - ## Ongoing Consequences
   - ## Story Progression
4. Remove redundant information and organize by importance
5. Keep the most recent/relevant information accessible

Respond with JSON: {"compressed_long_term": "...", "session_summary": "..."}"""


async def run(campaign_id: str, query: str, new_event: str) -> dict:
    """Returns {recalled_context, long_term_summary, recent_events}."""

    # 1. Load current memory state
    mem = await call_mcp(STATE_MCP_URL, "get_memory", {}, campaign_id)
    short_term: list[str] = mem.get("short_term", [])
    long_term: str = mem.get("long_term", "")

    # 2. Store new event if provided
    if new_event:
        turn_id = str(uuid.uuid4())
        await call_mcp(MEMORY_MCP_URL, "store", {"turn_id": turn_id, "text": new_event, "role": "dm"}, campaign_id)
        short_term.append(new_event)

    # 3. Semantic recall
    recall_result = await call_mcp(MEMORY_MCP_URL, "recall", {"query": query, "top_k": 8}, campaign_id)
    recalled_context: str = recall_result.get("context", "")

    # 4. Check if we should compress
    if len(short_term) >= STM_THRESHOLD:
        recent_events_text = "\n".join(short_term)
        try:
            decision = await call_llm_json([
                {"role": "system", "content": CUTOFF_SYSTEM},
                {"role": "user", "content": f"RECENT EVENTS:\n{recent_events_text}"},
            ])
            if decision.get("should_compress"):
                compression = await call_llm_json([
                    {"role": "system", "content": COMPRESSION_SYSTEM},
                    {"role": "user", "content": (
                        f"CURRENT LONG-TERM MEMORY:\n{long_term}\n\n"
                        f"SHORT-TERM MEMORY TO COMPRESS:\n{recent_events_text}"
                    )},
                ])
                long_term = compression.get("compressed_long_term", long_term)
                short_term = short_term[-3:]  # keep last 3 events
        except Exception:
            logger.warning("Memory compression failed (non-critical)", exc_info=True)

    # 5. Persist updated memory
    await call_mcp(STATE_MCP_URL, "update_memory", {"short_term": short_term, "long_term": long_term}, campaign_id)

    return {
        "recalled_context": recalled_context,
        "long_term_summary": long_term,
        "recent_events": short_term,
    }
