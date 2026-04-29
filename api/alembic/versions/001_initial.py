"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-04-28
"""

from typing import Sequence, Union

from alembic import op

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)')

    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_usage (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id     UUID REFERENCES users(id) ON DELETE SET NULL,
            provider    TEXT NOT NULL,
            model       TEXT NOT NULL,
            tokens_in   INT  NOT NULL,
            tokens_out  INT  NOT NULL,
            cached      BOOLEAN NOT NULL DEFAULT FALSE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute('CREATE INDEX IF NOT EXISTS idx_llm_usage_user ON llm_usage (user_id, created_at DESC)')

    op.execute("""
        DO $$ BEGIN
            CREATE TYPE campaign_phase AS ENUM (
                'character_creation',
                'campaign_design',
                'active',
                'completed'
            );
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id               UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title                 TEXT NOT NULL DEFAULT 'Untitled Campaign',
            language              TEXT NOT NULL DEFAULT 'en',
            phase                 campaign_phase NOT NULL DEFAULT 'character_creation',
            plan_json             JSONB,
            visual_style          TEXT,
            active_npc_id         UUID,
            active_npc_briefing   JSONB,
            active_npc_conv_start UUID,
            short_term_memory     JSONB NOT NULL DEFAULT '[]',
            long_term_memory      TEXT  NOT NULL DEFAULT '',
            created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute('CREATE INDEX IF NOT EXISTS idx_campaigns_user  ON campaigns (user_id)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_campaigns_phase ON campaigns (phase)')

    op.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            campaign_id         UUID NOT NULL UNIQUE REFERENCES campaigns(id) ON DELETE CASCADE,
            name                TEXT NOT NULL,
            background          TEXT NOT NULL,
            class_and_level     TEXT NOT NULL,
            abilities           TEXT[] NOT NULL DEFAULT '{}',
            equipment           TEXT[] NOT NULL DEFAULT '{}',
            limitations         TEXT[] NOT NULL DEFAULT '{}',
            power_level         TEXT NOT NULL,
            visual_description  TEXT NOT NULL,
            portrait_path       TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        DO $$ BEGIN
            CREATE TYPE turn_role AS ENUM ('player', 'dm', 'npc', 'system');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS npcs (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            campaign_id         UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            name                TEXT NOT NULL,
            role                TEXT NOT NULL,
            visual_description  TEXT NOT NULL,
            voice_id            TEXT NOT NULL,
            voice_instructions  TEXT NOT NULL,
            portrait_path       TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (campaign_id, name)
        )
    """)

    op.execute('CREATE INDEX IF NOT EXISTS idx_npcs_campaign ON npcs (campaign_id)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_npcs_name     ON npcs (campaign_id, name)')

    op.execute("""
        DO $$ BEGIN
            ALTER TABLE campaigns
                ADD CONSTRAINT fk_campaigns_active_npc
                FOREIGN KEY (active_npc_id) REFERENCES npcs(id)
                ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            role        turn_role NOT NULL,
            content     TEXT NOT NULL,
            npc_name    TEXT,
            audio_path  TEXT,
            image_path  TEXT,
            metadata    JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute('CREATE INDEX IF NOT EXISTS idx_turns_campaign_created ON turns (campaign_id, created_at DESC)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_turns_campaign_role    ON turns (campaign_id, role)')
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_turns_campaign_npc ON turns (campaign_id, npc_name)
        WHERE npc_name IS NOT NULL
    """)


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS turns')
    op.execute('ALTER TABLE campaigns DROP CONSTRAINT IF EXISTS fk_campaigns_active_npc')
    op.execute('DROP TABLE IF EXISTS npcs')
    op.execute('DROP TABLE IF EXISTS characters')
    op.execute('DROP TABLE IF EXISTS campaigns')
    op.execute('DROP TABLE IF EXISTS llm_usage')
    op.execute('DROP TABLE IF EXISTS users')
    op.execute('DROP TYPE IF EXISTS turn_role')
    op.execute('DROP TYPE IF EXISTS campaign_phase')
