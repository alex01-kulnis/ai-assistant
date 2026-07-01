"""add voice metadata to agent runs

Revision ID: 202607010001
Revises: 202605120001
Create Date: 2026-07-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202607010001"
down_revision: str | None = "202605120001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("input_mode", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("input_audio_path", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("input_transcript", sa.Text(), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("stt_provider", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("stt_model", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("stt_latency_ms", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "stt_latency_ms")
    op.drop_column("agent_runs", "stt_model")
    op.drop_column("agent_runs", "stt_provider")
    op.drop_column("agent_runs", "input_transcript")
    op.drop_column("agent_runs", "input_audio_path")
    op.drop_column("agent_runs", "input_mode")
