"""
Quarantine Manager — read-only isolation before termination.

Instead of immediately evicting a liable agent, moves it to a read-only
quarantine state where it can still be interrogated for forensic replay
but cannot issue new effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
import uuid


class QuarantineReason(str, Enum):
    """Why an agent was quarantined."""

    BEHAVIORAL_DRIFT = "behavioral_drift"
    LIABILITY_VIOLATION = "liability_violation"
    RING_BREACH = "ring_breach"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MANUAL = "manual"
    CASCADE_SLASH = "cascade_slash"


@dataclass
class QuarantineRecord:
    """Record of an agent in quarantine."""

    quarantine_id: str = field(default_factory=lambda: f"quar:{uuid.uuid4().hex[:8]}")
    agent_did: str = ""
    session_id: str = ""
    reason: QuarantineReason = QuarantineReason.MANUAL
    details: str = ""
    entered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    is_active: bool = True
    forensic_data: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def duration_seconds(self) -> float:
        end = self.released_at or datetime.now(timezone.utc)
        return (end - self.entered_at).total_seconds()


class QuarantineManager:
    """
    Manages agent quarantine — read-only isolation for forensic analysis.

    Quarantined agents:
    - Cannot write to VFS
    - Cannot execute saga steps
    - Cannot escalate ring level
    - CAN be queried for state/history (forensic replay)
    - Auto-released after timeout or manually released
    """

    DEFAULT_QUARANTINE_SECONDS = 300  # 5 minutes

    def __init__(self) -> None:
        self._quarantines: dict[str, QuarantineRecord] = {}

    def quarantine(
        self,
        agent_did: str,
        session_id: str,
        reason: QuarantineReason,
        details: str = "",
        duration_seconds: Optional[int] = None,
        forensic_data: Optional[dict] = None,
    ) -> QuarantineRecord:
        """
        Place an agent in quarantine.

        Args:
            agent_did: Agent to quarantine
            session_id: Session scope
            reason: Why the agent is being quarantined
            details: Additional details
            duration_seconds: How long to quarantine (None = indefinite)
            forensic_data: Any evidence/state to preserve

        Returns:
            QuarantineRecord
        """
        # Check if already quarantined
        existing = self.get_active_quarantine(agent_did, session_id)
        if existing:
            # Update existing quarantine with new reason
            existing.details += f"; escalated: {details}"
            if forensic_data:
                existing.forensic_data.update(forensic_data)
            return existing

        duration = duration_seconds or self.DEFAULT_QUARANTINE_SECONDS
        now = datetime.now(timezone.utc)

        record = QuarantineRecord(
            agent_did=agent_did,
            session_id=session_id,
            reason=reason,
            details=details,
            entered_at=now,
            expires_at=now + timedelta(seconds=duration) if duration else None,
            forensic_data=forensic_data or {},
        )
        self._quarantines[record.quarantine_id] = record
        return record

    def release(self, agent_did: str, session_id: str) -> Optional[QuarantineRecord]:
        """Release an agent from quarantine."""
        record = self.get_active_quarantine(agent_did, session_id)
        if record:
            record.is_active = False
            record.released_at = datetime.now(timezone.utc)
        return record

    def is_quarantined(self, agent_did: str, session_id: str) -> bool:
        """Check if an agent is currently quarantined."""
        record = self.get_active_quarantine(agent_did, session_id)
        return record is not None

    def get_active_quarantine(
        self, agent_did: str, session_id: str
    ) -> Optional[QuarantineRecord]:
        """Get the active quarantine record for an agent."""
        for record in self._quarantines.values():
            if (
                record.agent_did == agent_did
                and record.session_id == session_id
                and record.is_active
                and not record.is_expired
            ):
                return record
        return None

    def tick(self) -> list[QuarantineRecord]:
        """Check for expired quarantines and release them."""
        released = []
        for record in self._quarantines.values():
            if record.is_active and record.is_expired:
                record.is_active = False
                record.released_at = datetime.now(timezone.utc)
                released.append(record)
        return released

    def get_history(
        self, agent_did: Optional[str] = None, session_id: Optional[str] = None
    ) -> list[QuarantineRecord]:
        """Get quarantine history, optionally filtered."""
        records = list(self._quarantines.values())
        if agent_did:
            records = [r for r in records if r.agent_did == agent_did]
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        return records

    @property
    def active_quarantines(self) -> list[QuarantineRecord]:
        return [
            r for r in self._quarantines.values()
            if r.is_active and not r.is_expired
        ]

    @property
    def quarantine_count(self) -> int:
        return len(self.active_quarantines)
