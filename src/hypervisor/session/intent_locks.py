"""
Intent Locks — declare read/write/exclusive intent before execution.

Agents declare intent at saga step registration, allowing the hypervisor
to detect lock contention before execution. Includes deadlock detection
via wait-for graph analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class LockIntent(str, Enum):
    """Types of lock intent."""

    READ = "read"        # shared read access
    WRITE = "write"      # exclusive write access
    EXCLUSIVE = "exclusive"  # exclusive read + write


@dataclass
class IntentLock:
    """A declared intent lock on a resource."""

    lock_id: str = field(default_factory=lambda: f"lock:{uuid.uuid4().hex[:8]}")
    agent_did: str = ""
    session_id: str = ""
    resource_path: str = ""
    intent: LockIntent = LockIntent.READ
    acquired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    saga_step_id: Optional[str] = None


class LockContentionError(Exception):
    """Raised when lock contention is detected."""


class DeadlockError(Exception):
    """Raised when a deadlock is detected in the wait-for graph."""


class IntentLockManager:
    """
    Manages intent locks with contention detection and deadlock prevention.

    Lock compatibility matrix:
        READ    + READ      = OK (shared)
        READ    + WRITE     = CONTENTION
        READ    + EXCLUSIVE = CONTENTION
        WRITE   + WRITE     = CONTENTION
        WRITE   + EXCLUSIVE = CONTENTION
        EXCLUSIVE + any     = CONTENTION
    """

    def __init__(self) -> None:
        self._locks: dict[str, IntentLock] = {}
        # resource_path -> list of active lock_ids
        self._resource_locks: dict[str, list[str]] = {}
        # Wait-for graph: agent -> set of agents they're waiting on
        self._wait_for: dict[str, set[str]] = {}

    def acquire(
        self,
        agent_did: str,
        session_id: str,
        resource_path: str,
        intent: LockIntent,
        saga_step_id: Optional[str] = None,
    ) -> IntentLock:
        """
        Acquire an intent lock on a resource.

        Raises:
            LockContentionError: If the lock conflicts with existing locks
            DeadlockError: If acquiring would create a deadlock
        """
        # Check for contention
        conflicts = self._check_contention(resource_path, agent_did, intent)
        if conflicts:
            # Check for potential deadlock
            blocking_agents = {c.agent_did for c in conflicts}
            if self._would_deadlock(agent_did, blocking_agents):
                raise DeadlockError(
                    f"Deadlock detected: {agent_did} would wait on "
                    f"{blocking_agents} which are waiting on {agent_did}"
                )

            conflict_agents = ", ".join(c.agent_did for c in conflicts)
            raise LockContentionError(
                f"Lock contention on {resource_path}: "
                f"{agent_did} ({intent.value}) conflicts with {conflict_agents}"
            )

        lock = IntentLock(
            agent_did=agent_did,
            session_id=session_id,
            resource_path=resource_path,
            intent=intent,
            saga_step_id=saga_step_id,
        )
        self._locks[lock.lock_id] = lock
        self._resource_locks.setdefault(resource_path, []).append(lock.lock_id)
        return lock

    def release(self, lock_id: str) -> None:
        """Release a lock."""
        lock = self._locks.get(lock_id)
        if not lock:
            return
        lock.is_active = False
        resource_locks = self._resource_locks.get(lock.resource_path, [])
        if lock_id in resource_locks:
            resource_locks.remove(lock_id)
        # Clean up wait-for graph
        self._wait_for.pop(lock.agent_did, None)

    def release_agent_locks(self, agent_did: str, session_id: str) -> int:
        """Release all locks held by an agent in a session."""
        count = 0
        for lock in list(self._locks.values()):
            if lock.agent_did == agent_did and lock.session_id == session_id and lock.is_active:
                self.release(lock.lock_id)
                count += 1
        return count

    def release_session_locks(self, session_id: str) -> int:
        """Release all locks in a session."""
        count = 0
        for lock in list(self._locks.values()):
            if lock.session_id == session_id and lock.is_active:
                self.release(lock.lock_id)
                count += 1
        return count

    def get_agent_locks(self, agent_did: str, session_id: str) -> list[IntentLock]:
        """Get all active locks held by an agent."""
        return [
            l for l in self._locks.values()
            if l.agent_did == agent_did
            and l.session_id == session_id
            and l.is_active
        ]

    def get_resource_locks(self, resource_path: str) -> list[IntentLock]:
        """Get all active locks on a resource."""
        lock_ids = self._resource_locks.get(resource_path, [])
        return [
            self._locks[lid] for lid in lock_ids
            if lid in self._locks and self._locks[lid].is_active
        ]

    def _check_contention(
        self,
        resource_path: str,
        agent_did: str,
        intent: LockIntent,
    ) -> list[IntentLock]:
        """Check if the requested lock conflicts with existing locks."""
        conflicts = []
        for lock in self.get_resource_locks(resource_path):
            if lock.agent_did == agent_did:
                continue  # Same agent, no conflict
            if not self._is_compatible(lock.intent, intent):
                conflicts.append(lock)
        return conflicts

    @staticmethod
    def _is_compatible(existing: LockIntent, requested: LockIntent) -> bool:
        """Check if two lock intents are compatible."""
        if existing == LockIntent.READ and requested == LockIntent.READ:
            return True
        return False

    def _would_deadlock(self, agent_did: str, blocking_agents: set[str]) -> bool:
        """Check if waiting on blocking_agents would create a deadlock."""
        # Simple cycle detection via DFS
        visited: set[str] = set()
        stack = list(blocking_agents)

        while stack:
            current = stack.pop()
            if current == agent_did:
                return True
            if current in visited:
                continue
            visited.add(current)
            # Who is this agent waiting on?
            waiting_on = self._wait_for.get(current, set())
            stack.extend(waiting_on)

        return False

    @property
    def active_lock_count(self) -> int:
        return sum(1 for l in self._locks.values() if l.is_active)

    @property
    def contention_points(self) -> list[str]:
        """Resources with multiple active locks from different agents."""
        points = []
        for path, lock_ids in self._resource_locks.items():
            agents = {
                self._locks[lid].agent_did
                for lid in lock_ids
                if lid in self._locks and self._locks[lid].is_active
            }
            if len(agents) > 1:
                points.append(path)
        return points
