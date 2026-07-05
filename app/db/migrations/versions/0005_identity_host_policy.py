"""Identity & host policy: name-visibility policy, event type, host settings
toggles (spec 005), and pending-membership status for join approval.

Revision ID: 0005_identity_host_policy
Revises: 0004_event_cover
Create Date: 2026-07-05
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0005_identity_host_policy"
down_revision: Union[str, None] = "0004_event_cover"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "event",
        sa.Column("name_policy", sa.String(length=16), nullable=False, server_default="nobody"),
    )
    op.add_column("event", sa.Column("event_type", sa.String(length=16), nullable=True))
    op.add_column(
        "event",
        sa.Column(
            "gallery_visibility", sa.String(length=24), nullable=False, server_default="own_only"
        ),
    )
    op.add_column(
        "event",
        sa.Column("ai_edit_scope", sa.String(length=16), nullable=False, server_default="solo_only"),
    )
    op.add_column(
        "event",
        sa.Column("member_uploads", sa.String(length=16), nullable=False, server_default="enabled"),
    )
    op.add_column(
        "event",
        sa.Column("member_delete_own", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "event",
        sa.Column("join_approval", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "event",
        sa.Column("member_list_visible", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "membership",
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
    )


def downgrade() -> None:
    op.drop_column("membership", "status")
    op.drop_column("event", "member_list_visible")
    op.drop_column("event", "join_approval")
    op.drop_column("event", "member_delete_own")
    op.drop_column("event", "member_uploads")
    op.drop_column("event", "ai_edit_scope")
    op.drop_column("event", "gallery_visibility")
    op.drop_column("event", "event_type")
    op.drop_column("event", "name_policy")
