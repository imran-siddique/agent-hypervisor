"""Tests for dynamic ring elevation, breach detection, and ring inheritance."""

import pytest
from datetime import timedelta, datetime, timezone

from hypervisor.models import ExecutionRing
from hypervisor.rings.elevation import (
    RingElevationManager,
    RingElevation,
    RingElevationError,
)
from hypervisor.rings.breach_detector import (
    RingBreachDetector,
    BreachSeverity,
)


# ── Ring Elevation Tests ────────────────────────────────────────


class TestRingElevation:
    def test_request_elevation(self):
        mgr = RingElevationManager()
        elev = mgr.request_elevation(
            agent_did="a1",
            session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
            ttl_seconds=60,
            reason="Need write access",
        )
        assert elev.elevated_ring == ExecutionRing.RING_2_STANDARD
        assert elev.original_ring == ExecutionRing.RING_3_SANDBOX
        assert elev.is_active
        assert not elev.is_expired
        assert elev.remaining_seconds > 0

    def test_effective_ring_with_elevation(self):
        mgr = RingElevationManager()
        mgr.request_elevation(
            agent_did="a1", session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
            ttl_seconds=300,
        )
        effective = mgr.get_effective_ring("a1", "s1", ExecutionRing.RING_3_SANDBOX)
        assert effective == ExecutionRing.RING_2_STANDARD

    def test_effective_ring_without_elevation(self):
        mgr = RingElevationManager()
        effective = mgr.get_effective_ring("a1", "s1", ExecutionRing.RING_3_SANDBOX)
        assert effective == ExecutionRing.RING_3_SANDBOX

    def test_cannot_elevate_to_same_or_lower(self):
        mgr = RingElevationManager()
        with pytest.raises(RingElevationError):
            mgr.request_elevation(
                agent_did="a1", session_id="s1",
                current_ring=ExecutionRing.RING_2_STANDARD,
                target_ring=ExecutionRing.RING_3_SANDBOX,
            )

    def test_cannot_elevate_to_ring_0(self):
        mgr = RingElevationManager()
        with pytest.raises(RingElevationError, match="Ring 0"):
            mgr.request_elevation(
                agent_did="a1", session_id="s1",
                current_ring=ExecutionRing.RING_2_STANDARD,
                target_ring=ExecutionRing.RING_0_ROOT,
            )

    def test_duplicate_elevation_rejected(self):
        mgr = RingElevationManager()
        mgr.request_elevation(
            agent_did="a1", session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
            ttl_seconds=300,
        )
        with pytest.raises(RingElevationError, match="already has active"):
            mgr.request_elevation(
                agent_did="a1", session_id="s1",
                current_ring=ExecutionRing.RING_3_SANDBOX,
                target_ring=ExecutionRing.RING_2_STANDARD,
            )

    def test_revoke_elevation(self):
        mgr = RingElevationManager()
        elev = mgr.request_elevation(
            agent_did="a1", session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
            ttl_seconds=300,
        )
        mgr.revoke_elevation(elev.elevation_id)
        assert mgr.get_active_elevation("a1", "s1") is None

    def test_tick_expires_elevations(self):
        mgr = RingElevationManager()
        elev = mgr.request_elevation(
            agent_did="a1", session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
            ttl_seconds=1,
        )
        # Force expiry
        elev.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        expired = mgr.tick()
        assert len(expired) == 1
        assert not elev.is_active

    def test_ttl_capped_at_max(self):
        mgr = RingElevationManager()
        elev = mgr.request_elevation(
            agent_did="a1", session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
            ttl_seconds=999999,
        )
        max_ttl = RingElevationManager.MAX_ELEVATION_TTL
        assert elev.remaining_seconds <= max_ttl + 1

    def test_active_elevations_property(self):
        mgr = RingElevationManager()
        mgr.request_elevation(
            agent_did="a1", session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
        )
        mgr.request_elevation(
            agent_did="a2", session_id="s1",
            current_ring=ExecutionRing.RING_3_SANDBOX,
            target_ring=ExecutionRing.RING_2_STANDARD,
        )
        assert len(mgr.active_elevations) == 2


# ── Ring Inheritance Tests ──────────────────────────────────────


class TestRingInheritance:
    def test_child_inherits_parent_minus_one(self):
        mgr = RingElevationManager()
        child_ring = mgr.register_child(
            "parent", "child", ExecutionRing.RING_1_PRIVILEGED
        )
        assert child_ring == ExecutionRing.RING_2_STANDARD

    def test_child_of_sandbox_stays_sandbox(self):
        mgr = RingElevationManager()
        child_ring = mgr.register_child(
            "parent", "child", ExecutionRing.RING_3_SANDBOX
        )
        assert child_ring == ExecutionRing.RING_3_SANDBOX

    def test_child_of_ring2_gets_ring3(self):
        mgr = RingElevationManager()
        child_ring = mgr.register_child(
            "parent", "child", ExecutionRing.RING_2_STANDARD
        )
        assert child_ring == ExecutionRing.RING_3_SANDBOX

    def test_parent_child_tracking(self):
        mgr = RingElevationManager()
        mgr.register_child("p1", "c1", ExecutionRing.RING_1_PRIVILEGED)
        mgr.register_child("p1", "c2", ExecutionRing.RING_1_PRIVILEGED)
        assert mgr.get_parent("c1") == "p1"
        assert set(mgr.get_children("p1")) == {"c1", "c2"}

    def test_max_child_ring(self):
        mgr = RingElevationManager()
        assert mgr.get_max_child_ring(ExecutionRing.RING_0_ROOT) == ExecutionRing.RING_1_PRIVILEGED
        assert mgr.get_max_child_ring(ExecutionRing.RING_3_SANDBOX) == ExecutionRing.RING_3_SANDBOX


# ── Breach Detector Tests ──────────────────────────────────────


class TestBreachDetector:
    def test_no_breach_with_normal_pattern(self):
        detector = RingBreachDetector()
        for _ in range(10):
            result = detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_2_STANDARD,
                ExecutionRing.RING_2_STANDARD,
            )
        assert result is None

    def test_breach_detected_with_anomalous_calls(self):
        detector = RingBreachDetector()
        # Make 10 calls, all to more privileged ring
        # After circuit breaker trips, further calls return None (cooldown)
        # So collect all non-None results
        results = []
        for _ in range(10):
            result = detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_3_SANDBOX,
                ExecutionRing.RING_1_PRIVILEGED,
            )
            if result is not None:
                results.append(result)
        assert len(results) > 0
        assert results[-1].severity in (BreachSeverity.CRITICAL, BreachSeverity.HIGH)
        assert results[-1].anomaly_score > 0.5

    def test_circuit_breaker_tripped(self):
        detector = RingBreachDetector()
        for _ in range(10):
            detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_3_SANDBOX,
                ExecutionRing.RING_1_PRIVILEGED,
            )
        assert detector.is_breaker_tripped("a1", "s1")

    def test_breaker_not_tripped_for_normal(self):
        detector = RingBreachDetector()
        for _ in range(10):
            detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_2_STANDARD,
                ExecutionRing.RING_2_STANDARD,
            )
        assert not detector.is_breaker_tripped("a1", "s1")

    def test_reset_breaker(self):
        detector = RingBreachDetector()
        for _ in range(10):
            detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_3_SANDBOX,
                ExecutionRing.RING_1_PRIVILEGED,
            )
        detector.reset_breaker("a1", "s1")
        assert not detector.is_breaker_tripped("a1", "s1")

    def test_agent_stats(self):
        detector = RingBreachDetector()
        for _ in range(5):
            detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_2_STANDARD,
                ExecutionRing.RING_2_STANDARD,
            )
        stats = detector.get_agent_stats("a1", "s1")
        assert stats["total_calls"] == 5
        assert stats["window_calls"] == 5

    def test_stats_for_unknown_agent(self):
        detector = RingBreachDetector()
        stats = detector.get_agent_stats("unknown", "s1")
        assert stats["total_calls"] == 0

    def test_breach_history(self):
        detector = RingBreachDetector()
        for _ in range(10):
            detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_3_SANDBOX,
                ExecutionRing.RING_1_PRIVILEGED,
            )
        assert detector.breach_count > 0

    def test_mixed_call_pattern(self):
        detector = RingBreachDetector()
        # 3 normal + 7 anomalous = 70% anomaly
        for _ in range(3):
            detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_3_SANDBOX,
                ExecutionRing.RING_3_SANDBOX,
            )
        result = None
        for _ in range(7):
            result = detector.record_call(
                "a1", "s1",
                ExecutionRing.RING_3_SANDBOX,
                ExecutionRing.RING_1_PRIVILEGED,
            )
        assert result is not None
        assert result.severity in (BreachSeverity.HIGH, BreachSeverity.CRITICAL)
