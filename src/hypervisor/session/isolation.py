"""
Session Isolation Levels — configurable consistency per saga.

Analogous to database transaction isolation: low-stakes sagas
don't pay the coordination cost of high-stakes ones.
"""

from __future__ import annotations

from enum import Enum


class IsolationLevel(str, Enum):
    """Session isolation levels, analogous to database isolation."""

    SNAPSHOT = "snapshot"
    """
    Read from a consistent snapshot taken at saga start.
    Writes are buffered and applied at commit.
    Low coordination cost, suitable for read-heavy sagas.
    """

    READ_COMMITTED = "read_committed"
    """
    Reads see only committed writes from other agents.
    Each read gets the latest committed version.
    Moderate coordination cost.
    """

    SERIALIZABLE = "serializable"
    """
    Full serializable isolation. Reads and writes are ordered
    to produce a result equivalent to serial execution.
    Highest coordination cost, required for high-stakes sagas.
    """

    @property
    def requires_vector_clocks(self) -> bool:
        """Whether this isolation level needs vector clock enforcement."""
        return self in (IsolationLevel.READ_COMMITTED, IsolationLevel.SERIALIZABLE)

    @property
    def requires_intent_locks(self) -> bool:
        """Whether this isolation level needs intent lock enforcement."""
        return self == IsolationLevel.SERIALIZABLE

    @property
    def allows_concurrent_writes(self) -> bool:
        """Whether concurrent writes are allowed."""
        return self != IsolationLevel.SERIALIZABLE

    @property
    def coordination_cost(self) -> str:
        """Human-readable coordination cost."""
        if self == IsolationLevel.SNAPSHOT:
            return "low"
        elif self == IsolationLevel.READ_COMMITTED:
            return "moderate"
        return "high"
