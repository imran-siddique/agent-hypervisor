"""
Session-scoped Vector Clocks — causal consistency for shared state.

Each agent carries a vector clock. The VFS rejects writes that would
create causal violations and forces the conflicting agent to re-read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import copy


class CausalViolationError(Exception):
    """Raised when a write would violate causal ordering."""


@dataclass
class VectorClock:
    """A vector clock tracking causal ordering per agent."""

    clocks: dict[str, int] = field(default_factory=dict)

    def tick(self, agent_did: str) -> None:
        """Increment the clock for an agent (on write)."""
        self.clocks[agent_did] = self.clocks.get(agent_did, 0) + 1

    def get(self, agent_did: str) -> int:
        """Get current clock value for an agent."""
        return self.clocks.get(agent_did, 0)

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge two vector clocks (take component-wise max)."""
        merged = VectorClock(clocks=dict(self.clocks))
        for agent, clock in other.clocks.items():
            merged.clocks[agent] = max(merged.clocks.get(agent, 0), clock)
        return merged

    def happens_before(self, other: VectorClock) -> bool:
        """Check if self happens-before other (self < other)."""
        # self < other iff all(self[i] <= other[i]) and exists i where self[i] < other[i]
        all_agents = set(self.clocks.keys()) | set(other.clocks.keys())
        all_leq = all(
            self.clocks.get(a, 0) <= other.clocks.get(a, 0)
            for a in all_agents
        )
        any_lt = any(
            self.clocks.get(a, 0) < other.clocks.get(a, 0)
            for a in all_agents
        )
        return all_leq and any_lt

    def is_concurrent(self, other: VectorClock) -> bool:
        """Check if self and other are concurrent (neither happens-before the other)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def copy(self) -> VectorClock:
        return VectorClock(clocks=dict(self.clocks))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return False
        all_agents = set(self.clocks.keys()) | set(other.clocks.keys())
        return all(
            self.clocks.get(a, 0) == other.clocks.get(a, 0)
            for a in all_agents
        )


class VectorClockManager:
    """
    Manages vector clocks for a session's shared state.

    Each path in the VFS has an associated vector clock. On write,
    the agent must present a clock that is not concurrent with the
    path's current clock (unless they've read the latest version).
    """

    def __init__(self) -> None:
        # path -> current vector clock
        self._path_clocks: dict[str, VectorClock] = {}
        # agent -> their last-known vector clock
        self._agent_clocks: dict[str, VectorClock] = {}
        # Conflict counter
        self._conflict_count: int = 0

    def read(self, path: str, agent_did: str) -> VectorClock:
        """
        Record a read operation. Updates the agent's clock to include
        the path's current state.

        Returns the current vector clock for the path.
        """
        path_clock = self._path_clocks.get(path, VectorClock())
        agent_clock = self._agent_clocks.get(agent_did, VectorClock())

        # Merge: agent now knows about the path's state
        merged = agent_clock.merge(path_clock)
        self._agent_clocks[agent_did] = merged

        return path_clock.copy()

    def write(
        self,
        path: str,
        agent_did: str,
        strict: bool = True,
    ) -> VectorClock:
        """
        Record a write operation with causal consistency check.

        Args:
            path: The path being written
            agent_did: The writing agent
            strict: If True, reject concurrent writes (SERIALIZABLE).
                    If False, allow concurrent writes (EVENTUAL).

        Returns:
            The new vector clock for the path after the write.

        Raises:
            CausalViolationError: If the write would violate causal ordering
        """
        path_clock = self._path_clocks.get(path, VectorClock())
        agent_clock = self._agent_clocks.get(agent_did, VectorClock())

        # Check for causal violations
        if strict and path_clock.clocks:
            # Agent's clock must happen-after or equal to path's clock
            if path_clock.is_concurrent(agent_clock) or path_clock.happens_before(agent_clock):
                pass  # OK: agent has seen the latest state
            elif agent_clock.happens_before(path_clock):
                # Agent hasn't seen the latest state — causal violation
                self._conflict_count += 1
                raise CausalViolationError(
                    f"Agent {agent_did} has stale state for {path}. "
                    f"Agent clock: {agent_clock.clocks}, "
                    f"Path clock: {path_clock.clocks}. "
                    f"Must re-read before writing."
                )

        # Tick the agent's clock and update the path
        agent_clock.tick(agent_did)
        new_path_clock = path_clock.merge(agent_clock)
        self._path_clocks[path] = new_path_clock
        self._agent_clocks[agent_did] = agent_clock

        return new_path_clock

    def get_path_clock(self, path: str) -> VectorClock:
        """Get the current vector clock for a path."""
        return self._path_clocks.get(path, VectorClock()).copy()

    def get_agent_clock(self, agent_did: str) -> VectorClock:
        """Get an agent's current vector clock."""
        return self._agent_clocks.get(agent_did, VectorClock()).copy()

    @property
    def conflict_count(self) -> int:
        return self._conflict_count

    @property
    def tracked_paths(self) -> int:
        return len(self._path_clocks)
