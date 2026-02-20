"""
Dynamic Ring Elevation — time-bounded privilege escalation.

Like sudo with TTL: agents can request temporary ring elevation with
cryptographic attestation. Elevation auto-expires, preventing indefinite
privilege. Supports ring inheritance for spawned child agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

from hypervisor.models import ExecutionRing


@dataclass
class RingElevation:
    """A time-bounded ring elevation grant."""

    elevation_id: str = field(default_factory=lambda: f"elev:{uuid.uuid4().hex[:8]}")
    agent_did: str = ""
    session_id: str = ""
    original_ring: ExecutionRing = ExecutionRing.RING_3_SANDBOX
    elevated_ring: ExecutionRing = ExecutionRing.RING_2_STANDARD
    granted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attestation: Optional[str] = None
    reason: str = ""
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds())


class RingElevationManager:
    """
    Manages time-bounded ring elevations (like sudo with TTL).

    Agents request elevation with attestation. The manager grants
    time-limited elevation and auto-expires it. Child agents spawned
    during elevation inherit at most parent_ring - 1.
    """

    MAX_ELEVATION_TTL = 3600  # 1 hour max
    DEFAULT_TTL = 300  # 5 minutes

    def __init__(self) -> None:
        self._elevations: dict[str, RingElevation] = {}
        # Track parent-child relationships for ring inheritance
        self._parent_map: dict[str, str] = {}  # child_did -> parent_did
        self._children: dict[str, list[str]] = {}  # parent_did -> [child_dids]

    def request_elevation(
        self,
        agent_did: str,
        session_id: str,
        current_ring: ExecutionRing,
        target_ring: ExecutionRing,
        ttl_seconds: int = 0,
        attestation: Optional[str] = None,
        reason: str = "",
    ) -> RingElevation:
        """
        Request temporary ring elevation.

        Args:
            agent_did: Agent requesting elevation
            session_id: Session scope
            current_ring: Agent's current ring
            target_ring: Desired ring (must be more privileged)
            ttl_seconds: Time-to-live in seconds (0 = default)
            attestation: Cryptographic attestation or justification hash
            reason: Human-readable reason

        Raises:
            RingElevationError: If request is invalid
        """
        if target_ring.value >= current_ring.value:
            raise RingElevationError(
                f"Target ring {target_ring.value} is not more privileged "
                f"than current ring {current_ring.value}"
            )

        if target_ring == ExecutionRing.RING_0_ROOT:
            raise RingElevationError(
                "Ring 0 elevation not available via elevation manager — "
                "requires SRE Witness protocol"
            )

        # Check for existing active elevation
        existing = self.get_active_elevation(agent_did, session_id)
        if existing:
            raise RingElevationError(
                f"Agent {agent_did} already has active elevation "
                f"to ring {existing.elevated_ring.value}"
            )

        ttl = ttl_seconds if ttl_seconds > 0 else self.DEFAULT_TTL
        ttl = min(ttl, self.MAX_ELEVATION_TTL)

        now = datetime.now(timezone.utc)
        elevation = RingElevation(
            agent_did=agent_did,
            session_id=session_id,
            original_ring=current_ring,
            elevated_ring=target_ring,
            granted_at=now,
            expires_at=now + timedelta(seconds=ttl),
            attestation=attestation,
            reason=reason,
        )
        self._elevations[elevation.elevation_id] = elevation
        return elevation

    def get_active_elevation(
        self, agent_did: str, session_id: str
    ) -> Optional[RingElevation]:
        """Get the active (non-expired) elevation for an agent in a session."""
        for elev in self._elevations.values():
            if (
                elev.agent_did == agent_did
                and elev.session_id == session_id
                and elev.is_active
                and not elev.is_expired
            ):
                return elev
        return None

    def get_effective_ring(
        self, agent_did: str, session_id: str, base_ring: ExecutionRing
    ) -> ExecutionRing:
        """Get the effective ring considering any active elevation."""
        elev = self.get_active_elevation(agent_did, session_id)
        if elev and not elev.is_expired:
            return elev.elevated_ring
        return base_ring

    def revoke_elevation(self, elevation_id: str) -> None:
        """Manually revoke an elevation."""
        elev = self._elevations.get(elevation_id)
        if not elev:
            raise RingElevationError(f"Elevation {elevation_id} not found")
        elev.is_active = False

    def tick(self) -> list[RingElevation]:
        """
        Check for expired elevations and deactivate them.

        Returns list of newly expired elevations (for event bus notification).
        """
        expired = []
        for elev in self._elevations.values():
            if elev.is_active and elev.is_expired:
                elev.is_active = False
                expired.append(elev)
        return expired

    # -- Ring inheritance for spawned agents --

    def register_child(
        self, parent_did: str, child_did: str, parent_ring: ExecutionRing
    ) -> ExecutionRing:
        """
        Register a child agent spawned by a parent.

        Child inherits at most parent_ring - 1 (less privileged).
        Prevents privilege escalation via spawning.

        Returns:
            The ring assigned to the child
        """
        self._parent_map[child_did] = parent_did
        self._children.setdefault(parent_did, []).append(child_did)

        # Child gets at most one ring less privileged than parent
        child_ring_value = min(parent_ring.value + 1, ExecutionRing.RING_3_SANDBOX.value)
        return ExecutionRing(child_ring_value)

    def get_parent(self, child_did: str) -> Optional[str]:
        """Get the parent DID for a child agent."""
        return self._parent_map.get(child_did)

    def get_children(self, parent_did: str) -> list[str]:
        """Get all children spawned by a parent."""
        return list(self._children.get(parent_did, []))

    def get_max_child_ring(self, parent_ring: ExecutionRing) -> ExecutionRing:
        """Get the maximum ring a child can inherit."""
        child_value = min(parent_ring.value + 1, ExecutionRing.RING_3_SANDBOX.value)
        return ExecutionRing(child_value)

    @property
    def active_elevations(self) -> list[RingElevation]:
        return [e for e in self._elevations.values() if e.is_active and not e.is_expired]

    @property
    def elevation_count(self) -> int:
        return len(self._elevations)


class RingElevationError(Exception):
    """Raised for invalid ring elevation requests."""
