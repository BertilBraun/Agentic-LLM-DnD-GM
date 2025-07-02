from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from llm import llm_parse

# ───────────────────────────────────────────────────────────────
# Simplified Memory Models
# ───────────────────────────────────────────────────────────────


class CutoffDecision(BaseModel):
    reason: str = Field(description="Why this is or isn't a good cutoff point")
    should_compress: bool = Field(description='Whether this is a good point to compress memory')


class MemoryCompression(BaseModel):
    compressed_long_term: str = Field(description='Updated long-term memory in markdown format')
    session_summary: str = Field(description='Brief summary of what happened in the compressed session')


class SimpleMemorySystem(BaseModel):
    # Long-term memory as plain markdown text
    long_term_memory: str = Field(default='', description='Compressed campaign history in markdown')

    # Short-term memory as list of recent events
    short_term_memory: list[str] = Field(default_factory=list, description='Recent events')

    # Metadata
    compression_count: int = Field(default=0)
    last_compression: Optional[str] = Field(default=None)

    def add_event(self, event: str):
        """Add a new event to short-term memory"""
        self.short_term_memory.append(event.strip())

    async def check_for_cutoff(self) -> bool:
        """Ask LLM if this is a good point to compress memory"""
        if len(self.short_term_memory) < 5:  # Need minimum events
            return False

        recent_events = '\n'.join(self.short_term_memory)

        prompt = f"""Analyze the following recent D&D session events and determine if this is a good cutoff point for compressing memory.

RECENT EVENTS:
{recent_events}

Consider these factors:
- Have the players moved to a significantly different location/context?
- Has a major scene or encounter concluded?
- Are they transitioning between story beats (rest, travel, new chapter)?
- Have they left NPCs/locations they likely won't return to soon?
- Is there a natural narrative break?

A good cutoff point means the recent events form a cohesive "session" that can be summarized without losing important context for future interactions."""

        decision = llm_parse([{'role': 'user', 'content': prompt}], CutoffDecision)
        return decision.should_compress

    def compress_memory(self) -> bool:
        """Compress short-term memory into long-term memory using LLM"""
        if not self.short_term_memory:
            return False

        current_short_term = '\n'.join(self.short_term_memory)

        prompt = f"""You are helping manage memory for a D&D campaign. Compress the short-term memory into the long-term memory.

CURRENT LONG-TERM MEMORY:
{self.long_term_memory}

SHORT-TERM MEMORY TO COMPRESS:
{current_short_term}

Your task:
1. Update the long-term memory by integrating the short-term events
2. Preserve all important information: decisions, discoveries, relationships, quests, consequences
3. Use clear markdown formatting with sections like:
   - ## Key Decisions & Actions
   - ## Characters Met
   - ## Locations Visited  
   - ## Quests & Objectives
   - ## Important Information Learned
   - ## Ongoing Consequences
   - ## Story Progression

4. Remove redundant information and organize by importance
5. Keep the most recent/relevant information easily accessible
6. Maintain narrative flow and context

Format the response as updated long-term memory in markdown. Be comprehensive but concise."""

        try:
            compression = llm_parse([{'role': 'user', 'content': prompt}], MemoryCompression)

            # Update long-term memory
            self.long_term_memory = compression.compressed_long_term

            # Keep only the last 2-3 events for immediate context
            self.short_term_memory = self.short_term_memory[-3:]

            # Update metadata
            self.compression_count += 1
            self.last_compression = datetime.now().isoformat()

            print(f'📚 Memory compressed! Session summary: {compression.session_summary}')
            return True

        except Exception as e:
            print(f'⚠️ Memory compression failed: {e}')
            return False

    def get_full_context(self) -> str:
        """Get complete memory context for DM"""
        recent_events = '\n'.join(self.short_term_memory) if self.short_term_memory else 'No recent events.'

        if self.long_term_memory:
            return f"""# CAMPAIGN MEMORY

## Long-term Memory
{self.long_term_memory}

## Recent Events
{recent_events}
"""
        else:
            return f"""# CAMPAIGN MEMORY

## Recent Events  
{recent_events}
"""

    def get_player_summary(self) -> str:
        """Get a player-friendly summary of what they know"""
        return self.long_term_memory if self.long_term_memory else 'Campaign just started - no major events yet.'
