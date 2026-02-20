"""
Semantic Checkpoints — capture goals achieved, not just state.

Enables partial replay without re-running already-achieved effects.
Each checkpoint records what semantic goal was accomplished, allowing
the saga to skip completed goals on replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import hashlib
import uuid


@dataclass
class SemanticCheckpoint:
    """A checkpoint that captures what goal was achieved."""

    checkpoint_id: str = field(default_factory=lambda: f"ckpt:{uuid.uuid4().hex[:8]}")
    saga_id: str = ""
    step_id: str = ""
    goal_description: str = ""
    goal_hash: str = ""  # deterministic hash of the goal for dedup
    achieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    invalidated_reason: Optional[str] = None

    @staticmethod
    def compute_goal_hash(goal: str, step_id: str) -> str:
        """Compute deterministic hash for a goal."""
        content = f"{goal}:{step_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class CheckpointManager:
    """
    Manages semantic checkpoints for saga replay.

    Instead of replaying the entire saga on failure, the checkpoint
    manager allows skipping steps whose goals have already been achieved.

    Usage:
        mgr = CheckpointManager()
        mgr.save(saga_id, step_id, "Database schema migrated", {"version": 5})
        if mgr.is_achieved(saga_id, "Database schema migrated", step_id):
            skip_step()
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, list[SemanticCheckpoint]] = {}  # saga_id -> list
        self._by_goal_hash: dict[str, SemanticCheckpoint] = {}

    def save(
        self,
        saga_id: str,
        step_id: str,
        goal_description: str,
        state_snapshot: Optional[dict] = None,
    ) -> SemanticCheckpoint:
        """
        Save a semantic checkpoint.

        Args:
            saga_id: The saga this checkpoint belongs to
            step_id: The step that achieved the goal
            goal_description: Human-readable description of what was achieved
            state_snapshot: Optional state data to preserve

        Returns:
            SemanticCheckpoint
        """
        goal_hash = SemanticCheckpoint.compute_goal_hash(goal_description, step_id)

        checkpoint = SemanticCheckpoint(
            saga_id=saga_id,
            step_id=step_id,
            goal_description=goal_description,
            goal_hash=goal_hash,
            state_snapshot=state_snapshot or {},
        )

        self._checkpoints.setdefault(saga_id, []).append(checkpoint)
        self._by_goal_hash[goal_hash] = checkpoint
        return checkpoint

    def is_achieved(
        self,
        saga_id: str,
        goal_description: str,
        step_id: str,
    ) -> bool:
        """Check if a goal has already been achieved (for skip-on-replay)."""
        goal_hash = SemanticCheckpoint.compute_goal_hash(goal_description, step_id)
        checkpoint = self._by_goal_hash.get(goal_hash)
        return (
            checkpoint is not None
            and checkpoint.saga_id == saga_id
            and checkpoint.is_valid
        )

    def get_checkpoint(
        self,
        saga_id: str,
        goal_description: str,
        step_id: str,
    ) -> Optional[SemanticCheckpoint]:
        """Get a specific checkpoint if it exists and is valid."""
        goal_hash = SemanticCheckpoint.compute_goal_hash(goal_description, step_id)
        checkpoint = self._by_goal_hash.get(goal_hash)
        if checkpoint and checkpoint.saga_id == saga_id and checkpoint.is_valid:
            return checkpoint
        return None

    def invalidate(
        self,
        saga_id: str,
        step_id: str,
        reason: str = "",
    ) -> int:
        """
        Invalidate checkpoints for a step (e.g., if state changed).

        Returns count of invalidated checkpoints.
        """
        count = 0
        for ckpt in self._checkpoints.get(saga_id, []):
            if ckpt.step_id == step_id and ckpt.is_valid:
                ckpt.is_valid = False
                ckpt.invalidated_reason = reason
                count += 1
        return count

    def get_saga_checkpoints(self, saga_id: str) -> list[SemanticCheckpoint]:
        """Get all valid checkpoints for a saga."""
        return [
            c for c in self._checkpoints.get(saga_id, [])
            if c.is_valid
        ]

    def get_replay_plan(self, saga_id: str, steps: list[str]) -> list[str]:
        """
        Given a list of step_ids, return which ones need to be re-executed
        (i.e., don't have valid checkpoints).

        Returns list of step_ids that need execution.
        """
        achieved = {c.step_id for c in self.get_saga_checkpoints(saga_id)}
        return [s for s in steps if s not in achieved]

    @property
    def total_checkpoints(self) -> int:
        return sum(len(v) for v in self._checkpoints.values())

    @property
    def valid_checkpoints(self) -> int:
        return sum(
            1 for saga_ckpts in self._checkpoints.values()
            for c in saga_ckpts if c.is_valid
        )
