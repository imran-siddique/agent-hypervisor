"""
Persistent Liability Ledger — per-agent history of all liability events.

Stores vouches, slashes, quarantines, fault scores as a persistent ledger.
Queryable for admission decisions: should this agent be admitted to a new saga?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class LedgerEntryType(str, Enum):
    """Types of liability ledger entries."""

    VOUCH_GIVEN = "vouch_given"
    VOUCH_RECEIVED = "vouch_received"
    VOUCH_RELEASED = "vouch_released"
    SLASH_RECEIVED = "slash_received"
    SLASH_CASCADED = "slash_cascaded"
    QUARANTINE_ENTERED = "quarantine_entered"
    QUARANTINE_RELEASED = "quarantine_released"
    FAULT_ATTRIBUTED = "fault_attributed"
    CLEAN_SESSION = "clean_session"  # session completed without incident


@dataclass
class LedgerEntry:
    """A single entry in the liability ledger."""

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_did: str = ""
    entry_type: LedgerEntryType = LedgerEntryType.CLEAN_SESSION
    session_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: float = 0.0  # 0.0–1.0, how severe the event was
    details: str = ""
    related_agent: Optional[str] = None  # other agent involved


@dataclass
class AgentRiskProfile:
    """Computed risk profile for an agent based on ledger history."""

    agent_did: str
    total_entries: int = 0
    slash_count: int = 0
    quarantine_count: int = 0
    clean_session_count: int = 0
    fault_score_avg: float = 0.0
    risk_score: float = 0.0  # 0.0 = safe, 1.0 = maximum risk
    recommendation: str = "admit"  # "admit", "probation", "deny"


class LiabilityLedger:
    """
    Persistent per-agent liability history.

    Tracks all liability events across sessions. Used for:
    - Admission decisions (should we let this agent into a new saga?)
    - Risk scoring (how reliable is this agent historically?)
    - Audit trail (complete liability history for compliance)
    """

    # Risk thresholds
    PROBATION_THRESHOLD = 0.3
    DENY_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._by_agent: dict[str, list[LedgerEntry]] = {}

    def record(
        self,
        agent_did: str,
        entry_type: LedgerEntryType,
        session_id: str = "",
        severity: float = 0.0,
        details: str = "",
        related_agent: Optional[str] = None,
    ) -> LedgerEntry:
        """Record a liability event."""
        entry = LedgerEntry(
            agent_did=agent_did,
            entry_type=entry_type,
            session_id=session_id,
            severity=severity,
            details=details,
            related_agent=related_agent,
        )
        self._entries.append(entry)
        self._by_agent.setdefault(agent_did, []).append(entry)
        return entry

    def get_agent_history(self, agent_did: str) -> list[LedgerEntry]:
        """Get all ledger entries for an agent."""
        return list(self._by_agent.get(agent_did, []))

    def compute_risk_profile(self, agent_did: str) -> AgentRiskProfile:
        """
        Compute a risk profile for an agent based on their ledger history.

        Risk score formula:
        - Each slash adds 0.15 * severity
        - Each quarantine adds 0.10 * severity
        - Each fault attribution adds 0.05 * severity
        - Each clean session subtracts 0.05
        - Clamped to [0.0, 1.0]
        """
        entries = self.get_agent_history(agent_did)
        if not entries:
            return AgentRiskProfile(agent_did=agent_did, recommendation="admit")

        slash_count = 0
        quarantine_count = 0
        clean_count = 0
        fault_scores = []
        risk = 0.0

        for entry in entries:
            if entry.entry_type in (LedgerEntryType.SLASH_RECEIVED, LedgerEntryType.SLASH_CASCADED):
                slash_count += 1
                risk += 0.15 * max(entry.severity, 0.5)
            elif entry.entry_type == LedgerEntryType.QUARANTINE_ENTERED:
                quarantine_count += 1
                risk += 0.10 * max(entry.severity, 0.3)
            elif entry.entry_type == LedgerEntryType.FAULT_ATTRIBUTED:
                fault_scores.append(entry.severity)
                risk += 0.05 * entry.severity
            elif entry.entry_type == LedgerEntryType.CLEAN_SESSION:
                clean_count += 1
                risk -= 0.05

        risk = max(0.0, min(1.0, risk))
        avg_fault = sum(fault_scores) / len(fault_scores) if fault_scores else 0.0

        if risk >= self.DENY_THRESHOLD:
            recommendation = "deny"
        elif risk >= self.PROBATION_THRESHOLD:
            recommendation = "probation"
        else:
            recommendation = "admit"

        return AgentRiskProfile(
            agent_did=agent_did,
            total_entries=len(entries),
            slash_count=slash_count,
            quarantine_count=quarantine_count,
            clean_session_count=clean_count,
            fault_score_avg=round(avg_fault, 4),
            risk_score=round(risk, 4),
            recommendation=recommendation,
        )

    def should_admit(self, agent_did: str) -> tuple[bool, str]:
        """
        Quick admission check based on historical risk.

        Returns:
            (should_admit, reason)
        """
        profile = self.compute_risk_profile(agent_did)
        if profile.recommendation == "deny":
            return False, f"Risk score {profile.risk_score:.2f} exceeds threshold"
        return True, profile.recommendation

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @property
    def tracked_agents(self) -> list[str]:
        return list(self._by_agent.keys())
