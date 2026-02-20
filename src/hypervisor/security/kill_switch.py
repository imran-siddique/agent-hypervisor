"""
Kill Switch — graceful agent termination with saga handoff.

When an agent is forcibly terminated, attempts to hand off in-flight
saga steps to a substitute agent before triggering compensation.
Reduces false-positive rollbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class KillReason(str, Enum):
    """Why an agent was killed."""

    BEHAVIORAL_DRIFT = "behavioral_drift"
    RATE_LIMIT = "rate_limit"
    RING_BREACH = "ring_breach"
    MANUAL = "manual"
    QUARANTINE_TIMEOUT = "quarantine_timeout"
    SESSION_TIMEOUT = "session_timeout"


class HandoffStatus(str, Enum):
    """Status of a saga step handoff."""

    PENDING = "pending"
    HANDED_OFF = "handed_off"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class StepHandoff:
    """A saga step being handed off from a killed agent."""

    step_id: str
    saga_id: str
    from_agent: str
    to_agent: Optional[str] = None
    status: HandoffStatus = HandoffStatus.PENDING


@dataclass
class KillResult:
    """Result of a kill switch operation."""

    kill_id: str = field(default_factory=lambda: f"kill:{uuid.uuid4().hex[:8]}")
    agent_did: str = ""
    session_id: str = ""
    reason: KillReason = KillReason.MANUAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    handoffs: list[StepHandoff] = field(default_factory=list)
    handoff_success_count: int = 0
    compensation_triggered: bool = False
    details: str = ""


class KillSwitch:
    """
    Manages graceful agent termination with saga step handoff.

    When killing an agent:
    1. Quarantine the agent (read-only)
    2. Identify in-flight saga steps assigned to the agent
    3. Attempt to hand off each step to a substitute agent
    4. If no substitute available, trigger compensation
    5. Record the kill in the audit trail
    """

    def __init__(self) -> None:
        self._kill_history: list[KillResult] = []
        # Available substitutes per session: session_id -> [agent_dids]
        self._substitutes: dict[str, list[str]] = {}

    def register_substitute(
        self, session_id: str, agent_did: str
    ) -> None:
        """Register an agent as available for step handoff."""
        self._substitutes.setdefault(session_id, []).append(agent_did)

    def unregister_substitute(
        self, session_id: str, agent_did: str
    ) -> None:
        """Remove an agent from the substitute pool."""
        subs = self._substitutes.get(session_id, [])
        if agent_did in subs:
            subs.remove(agent_did)

    def kill(
        self,
        agent_did: str,
        session_id: str,
        reason: KillReason,
        in_flight_steps: Optional[list[dict]] = None,
        details: str = "",
    ) -> KillResult:
        """
        Kill an agent with graceful saga handoff.

        Args:
            agent_did: Agent to kill
            session_id: Session scope
            reason: Why the agent is being killed
            in_flight_steps: List of {step_id, saga_id} for in-flight steps
            details: Additional context

        Returns:
            KillResult with handoff outcomes
        """
        in_flight = in_flight_steps or []

        handoffs = []
        handoff_success = 0

        for step_info in in_flight:
            step_id = step_info.get("step_id", "")
            saga_id = step_info.get("saga_id", "")

            handoff = StepHandoff(
                step_id=step_id,
                saga_id=saga_id,
                from_agent=agent_did,
            )

            # Try to find a substitute
            substitute = self._find_substitute(session_id, agent_did)
            if substitute:
                handoff.to_agent = substitute
                handoff.status = HandoffStatus.HANDED_OFF
                handoff_success += 1
            else:
                handoff.status = HandoffStatus.COMPENSATED

            handoffs.append(handoff)

        result = KillResult(
            agent_did=agent_did,
            session_id=session_id,
            reason=reason,
            handoffs=handoffs,
            handoff_success_count=handoff_success,
            compensation_triggered=any(
                h.status == HandoffStatus.COMPENSATED for h in handoffs
            ),
            details=details,
        )
        self._kill_history.append(result)

        # Remove killed agent from substitute pool
        self.unregister_substitute(session_id, agent_did)

        return result

    def _find_substitute(
        self, session_id: str, exclude_did: str
    ) -> Optional[str]:
        """Find an available substitute agent."""
        subs = self._substitutes.get(session_id, [])
        for agent in subs:
            if agent != exclude_did:
                return agent
        return None

    @property
    def kill_history(self) -> list[KillResult]:
        return list(self._kill_history)

    @property
    def total_kills(self) -> int:
        return len(self._kill_history)

    @property
    def total_handoffs(self) -> int:
        return sum(r.handoff_success_count for r in self._kill_history)
