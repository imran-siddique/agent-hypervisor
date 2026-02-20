"""
Ring Breach Detector — anomaly scoring for ring call patterns.

Monitors per-agent ring call patterns using a sliding window. If an
inner-ring agent begins issuing outer-ring calls at unusual frequency,
triggers a circuit breaker before the full Joint Liability model kicks in.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from hypervisor.models import ExecutionRing


class BreachSeverity(str, Enum):
    """Severity levels for ring breach anomalies."""

    NONE = "none"
    LOW = "low"        # unusual but within tolerance
    MEDIUM = "medium"  # pattern anomaly — trigger warning
    HIGH = "high"      # circuit breaker recommended
    CRITICAL = "critical"  # immediate demotion recommended


@dataclass
class BreachEvent:
    """A detected ring breach anomaly."""

    agent_did: str
    session_id: str
    severity: BreachSeverity
    anomaly_score: float
    call_count_window: int
    expected_rate: float
    actual_rate: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


@dataclass
class AgentCallProfile:
    """Sliding window of ring call patterns for an agent."""

    agent_did: str
    session_id: str
    calls: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_calls: int = 0
    ring_call_counts: dict = field(default_factory=lambda: defaultdict(int))
    breaker_tripped: bool = False
    breaker_tripped_at: Optional[datetime] = None


class RingBreachDetector:
    """
    Detects anomalous ring call patterns using sliding window analysis.

    For each agent, tracks which rings they're calling into over a time
    window. If the call pattern deviates from their expected ring level,
    triggers breach events with severity scoring.
    """

    WINDOW_SECONDS = 60
    LOW_THRESHOLD = 0.3     # 30% anomalous calls
    MEDIUM_THRESHOLD = 0.5  # 50%
    HIGH_THRESHOLD = 0.7    # 70%
    CRITICAL_THRESHOLD = 0.9  # 90%
    CIRCUIT_BREAKER_COOLDOWN = 30  # seconds

    def __init__(self, window_seconds: int = 0) -> None:
        self._profiles: dict[str, AgentCallProfile] = {}
        self._breach_history: list[BreachEvent] = []
        self.window_seconds = window_seconds or self.WINDOW_SECONDS

    def record_call(
        self,
        agent_did: str,
        session_id: str,
        agent_ring: ExecutionRing,
        called_ring: ExecutionRing,
    ) -> Optional[BreachEvent]:
        """
        Record a ring call and check for anomalies.

        Returns a BreachEvent if anomaly detected, None otherwise.
        """
        key = f"{agent_did}:{session_id}"
        profile = self._profiles.get(key)
        if not profile:
            profile = AgentCallProfile(agent_did=agent_did, session_id=session_id)
            self._profiles[key] = profile

        now = datetime.now(timezone.utc)
        profile.calls.append((now, agent_ring, called_ring))
        profile.total_calls += 1
        profile.ring_call_counts[called_ring.value] += 1

        # Prune old calls outside window
        cutoff = now - timedelta(seconds=self.window_seconds)
        while profile.calls and profile.calls[0][0] < cutoff:
            profile.calls.popleft()

        # Check circuit breaker cooldown
        if profile.breaker_tripped and profile.breaker_tripped_at:
            cooldown_end = profile.breaker_tripped_at + timedelta(
                seconds=self.CIRCUIT_BREAKER_COOLDOWN
            )
            if now < cooldown_end:
                return None  # Still in cooldown

        # Compute anomaly score
        return self._analyze(profile, agent_ring, now)

    def _analyze(
        self,
        profile: AgentCallProfile,
        agent_ring: ExecutionRing,
        now: datetime,
    ) -> Optional[BreachEvent]:
        """Analyze call pattern for anomalies."""
        if len(profile.calls) < 5:
            return None  # Not enough data

        # Count calls to rings more privileged than the agent's ring
        anomalous = sum(
            1 for _, _, called in profile.calls
            if called.value < agent_ring.value  # lower value = more privileged
        )

        total = len(profile.calls)
        anomaly_rate = anomalous / total if total > 0 else 0.0

        # Determine severity
        if anomaly_rate >= self.CRITICAL_THRESHOLD:
            severity = BreachSeverity.CRITICAL
        elif anomaly_rate >= self.HIGH_THRESHOLD:
            severity = BreachSeverity.HIGH
        elif anomaly_rate >= self.MEDIUM_THRESHOLD:
            severity = BreachSeverity.MEDIUM
        elif anomaly_rate >= self.LOW_THRESHOLD:
            severity = BreachSeverity.LOW
        else:
            return None

        # Trip circuit breaker on HIGH or CRITICAL
        if severity in (BreachSeverity.HIGH, BreachSeverity.CRITICAL):
            profile.breaker_tripped = True
            profile.breaker_tripped_at = now

        event = BreachEvent(
            agent_did=profile.agent_did,
            session_id=profile.session_id,
            severity=severity,
            anomaly_score=anomaly_rate,
            call_count_window=total,
            expected_rate=0.0,
            actual_rate=anomaly_rate,
            details=(
                f"{anomalous}/{total} calls to more-privileged rings "
                f"in {self.window_seconds}s window"
            ),
        )
        self._breach_history.append(event)
        return event

    def is_breaker_tripped(self, agent_did: str, session_id: str) -> bool:
        """Check if the circuit breaker is tripped for an agent."""
        key = f"{agent_did}:{session_id}"
        profile = self._profiles.get(key)
        if not profile or not profile.breaker_tripped:
            return False

        # Check cooldown
        if profile.breaker_tripped_at:
            cooldown_end = profile.breaker_tripped_at + timedelta(
                seconds=self.CIRCUIT_BREAKER_COOLDOWN
            )
            if datetime.now(timezone.utc) >= cooldown_end:
                profile.breaker_tripped = False
                return False

        return True

    def reset_breaker(self, agent_did: str, session_id: str) -> None:
        """Manually reset the circuit breaker for an agent."""
        key = f"{agent_did}:{session_id}"
        profile = self._profiles.get(key)
        if profile:
            profile.breaker_tripped = False
            profile.breaker_tripped_at = None

    def get_agent_stats(
        self, agent_did: str, session_id: str
    ) -> dict:
        """Get call statistics for an agent."""
        key = f"{agent_did}:{session_id}"
        profile = self._profiles.get(key)
        if not profile:
            return {"total_calls": 0, "window_calls": 0, "breaker_tripped": False}

        return {
            "total_calls": profile.total_calls,
            "window_calls": len(profile.calls),
            "breaker_tripped": profile.breaker_tripped,
            "ring_distribution": dict(profile.ring_call_counts),
        }

    @property
    def breach_history(self) -> list[BreachEvent]:
        return list(self._breach_history)

    @property
    def breach_count(self) -> int:
        return len(self._breach_history)
