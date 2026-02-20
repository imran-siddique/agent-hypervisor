"""Tests for vector clocks, intent locks, isolation levels, rate limiter, and kill switch."""

import pytest
from datetime import timedelta, datetime, timezone

from hypervisor.models import ExecutionRing
from hypervisor.session.vector_clock import (
    VectorClock,
    VectorClockManager,
    CausalViolationError,
)
from hypervisor.session.intent_locks import (
    IntentLockManager,
    LockIntent,
    LockContentionError,
    DeadlockError,
)
from hypervisor.session.isolation import IsolationLevel
from hypervisor.security.rate_limiter import (
    AgentRateLimiter,
    RateLimitExceeded,
    TokenBucket,
)
from hypervisor.security.kill_switch import (
    KillSwitch,
    KillReason,
    HandoffStatus,
)


# ── Vector Clock Tests ──────────────────────────────────────────


class TestVectorClock:
    def test_tick(self):
        vc = VectorClock()
        vc.tick("a1")
        assert vc.get("a1") == 1
        vc.tick("a1")
        assert vc.get("a1") == 2

    def test_merge(self):
        vc1 = VectorClock(clocks={"a1": 3, "a2": 1})
        vc2 = VectorClock(clocks={"a1": 1, "a2": 5})
        merged = vc1.merge(vc2)
        assert merged.get("a1") == 3
        assert merged.get("a2") == 5

    def test_happens_before(self):
        vc1 = VectorClock(clocks={"a1": 1, "a2": 1})
        vc2 = VectorClock(clocks={"a1": 2, "a2": 1})
        assert vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)

    def test_concurrent(self):
        vc1 = VectorClock(clocks={"a1": 2, "a2": 1})
        vc2 = VectorClock(clocks={"a1": 1, "a2": 2})
        assert vc1.is_concurrent(vc2)

    def test_equal(self):
        vc1 = VectorClock(clocks={"a1": 1, "a2": 2})
        vc2 = VectorClock(clocks={"a1": 1, "a2": 2})
        assert vc1 == vc2

    def test_not_equal(self):
        vc1 = VectorClock(clocks={"a1": 1})
        vc2 = VectorClock(clocks={"a1": 2})
        assert vc1 != vc2

    def test_copy(self):
        vc = VectorClock(clocks={"a1": 1})
        copy = vc.copy()
        copy.tick("a1")
        assert vc.get("a1") == 1  # original unchanged


class TestVectorClockManager:
    def test_read_updates_agent_clock(self):
        mgr = VectorClockManager()
        mgr.write("/data/file1", "a1")
        mgr.read("/data/file1", "a2")
        agent_clock = mgr.get_agent_clock("a2")
        assert agent_clock.get("a1") == 1

    def test_write_advances_path_clock(self):
        mgr = VectorClockManager()
        mgr.write("/data/file1", "a1")
        path_clock = mgr.get_path_clock("/data/file1")
        assert path_clock.get("a1") == 1

    def test_causal_violation_detected(self):
        mgr = VectorClockManager()
        mgr.write("/data/file1", "a1")
        mgr.write("/data/file1", "a1")  # a1 advances to 2
        # a2 never read the file — has stale state → violation
        with pytest.raises(CausalViolationError):
            mgr.write("/data/file1", "a2", strict=True)

    def test_read_then_write_no_violation(self):
        mgr = VectorClockManager()
        mgr.write("/data/file1", "a1")
        mgr.read("/data/file1", "a2")  # a2 catches up
        mgr.write("/data/file1", "a2", strict=True)  # OK: a2 has seen latest

    def test_non_strict_allows_concurrent(self):
        mgr = VectorClockManager()
        mgr.write("/data/file1", "a1", strict=False)
        mgr.write("/data/file1", "a2", strict=False)
        assert mgr.tracked_paths == 1

    def test_conflict_count(self):
        mgr = VectorClockManager()
        assert mgr.conflict_count == 0


# ── Intent Lock Tests ───────────────────────────────────────────


class TestIntentLocks:
    def test_acquire_read_locks(self):
        mgr = IntentLockManager()
        l1 = mgr.acquire("a1", "s1", "/data/file", LockIntent.READ)
        l2 = mgr.acquire("a2", "s1", "/data/file", LockIntent.READ)
        assert l1.is_active
        assert l2.is_active

    def test_write_conflicts_with_read(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/data/file", LockIntent.READ)
        with pytest.raises(LockContentionError):
            mgr.acquire("a2", "s1", "/data/file", LockIntent.WRITE)

    def test_write_conflicts_with_write(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/data/file", LockIntent.WRITE)
        with pytest.raises(LockContentionError):
            mgr.acquire("a2", "s1", "/data/file", LockIntent.WRITE)

    def test_exclusive_conflicts_with_read(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/data/file", LockIntent.READ)
        with pytest.raises(LockContentionError):
            mgr.acquire("a2", "s1", "/data/file", LockIntent.EXCLUSIVE)

    def test_same_agent_no_conflict(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/data/file", LockIntent.WRITE)
        # Same agent, different lock type — allowed
        mgr.acquire("a1", "s1", "/data/file", LockIntent.READ)

    def test_release_lock(self):
        mgr = IntentLockManager()
        lock = mgr.acquire("a1", "s1", "/data/file", LockIntent.WRITE)
        mgr.release(lock.lock_id)
        # Now another agent can acquire
        mgr.acquire("a2", "s1", "/data/file", LockIntent.WRITE)

    def test_release_agent_locks(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/file1", LockIntent.WRITE)
        mgr.acquire("a1", "s1", "/file2", LockIntent.EXCLUSIVE)
        count = mgr.release_agent_locks("a1", "s1")
        assert count == 2
        assert mgr.active_lock_count == 0

    def test_release_session_locks(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/file1", LockIntent.READ)
        mgr.acquire("a2", "s1", "/file2", LockIntent.WRITE)
        count = mgr.release_session_locks("s1")
        assert count == 2

    def test_contention_points(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/shared", LockIntent.READ)
        mgr.acquire("a2", "s1", "/shared", LockIntent.READ)
        points = mgr.contention_points
        assert "/shared" in points

    def test_deadlock_detection(self):
        mgr = IntentLockManager()
        # a1 holds lock on /f1
        mgr.acquire("a1", "s1", "/f1", LockIntent.WRITE)
        # a2 holds lock on /f2
        mgr.acquire("a2", "s1", "/f2", LockIntent.WRITE)
        # Simulate wait-for: a1 waiting on a2
        mgr._wait_for["a1"] = {"a2"}
        # a2 trying to acquire /f1 would create cycle
        with pytest.raises(DeadlockError):
            mgr.acquire("a2", "s1", "/f1", LockIntent.WRITE)

    def test_get_agent_locks(self):
        mgr = IntentLockManager()
        mgr.acquire("a1", "s1", "/f1", LockIntent.READ)
        mgr.acquire("a1", "s1", "/f2", LockIntent.WRITE)
        locks = mgr.get_agent_locks("a1", "s1")
        assert len(locks) == 2


# ── Isolation Level Tests ──────────────────────────────────────


class TestIsolationLevels:
    def test_snapshot_properties(self):
        level = IsolationLevel.SNAPSHOT
        assert not level.requires_vector_clocks
        assert not level.requires_intent_locks
        assert level.allows_concurrent_writes
        assert level.coordination_cost == "low"

    def test_read_committed_properties(self):
        level = IsolationLevel.READ_COMMITTED
        assert level.requires_vector_clocks
        assert not level.requires_intent_locks
        assert level.allows_concurrent_writes
        assert level.coordination_cost == "moderate"

    def test_serializable_properties(self):
        level = IsolationLevel.SERIALIZABLE
        assert level.requires_vector_clocks
        assert level.requires_intent_locks
        assert not level.allows_concurrent_writes
        assert level.coordination_cost == "high"


# ── Rate Limiter Tests ──────────────────────────────────────────


class TestRateLimiter:
    def test_allow_under_limit(self):
        limiter = AgentRateLimiter()
        assert limiter.check("a1", "s1", ExecutionRing.RING_2_STANDARD)

    def test_reject_over_limit(self):
        limiter = AgentRateLimiter()
        # Sandbox has capacity 10 — exhaust it
        for _ in range(10):
            limiter.try_check("a1", "s1", ExecutionRing.RING_3_SANDBOX)
        # 11th should fail
        assert not limiter.try_check("a1", "s1", ExecutionRing.RING_3_SANDBOX)

    def test_exception_on_limit(self):
        limiter = AgentRateLimiter()
        for _ in range(10):
            limiter.check("a1", "s1", ExecutionRing.RING_3_SANDBOX)
        with pytest.raises(RateLimitExceeded):
            limiter.check("a1", "s1", ExecutionRing.RING_3_SANDBOX)

    def test_different_rings_different_limits(self):
        limiter = AgentRateLimiter()
        # Ring 0 has much higher capacity than Ring 3
        for _ in range(50):
            assert limiter.try_check("a1", "s1", ExecutionRing.RING_0_ROOT)

    def test_update_ring(self):
        limiter = AgentRateLimiter()
        limiter.update_ring("a1", "s1", ExecutionRing.RING_0_ROOT)
        # Should have generous limits now
        for _ in range(100):
            assert limiter.try_check("a1", "s1", ExecutionRing.RING_0_ROOT)

    def test_stats(self):
        limiter = AgentRateLimiter()
        limiter.check("a1", "s1", ExecutionRing.RING_2_STANDARD)
        limiter.check("a1", "s1", ExecutionRing.RING_2_STANDARD)
        stats = limiter.get_stats("a1", "s1")
        assert stats is not None
        assert stats.total_requests == 2
        assert stats.rejected_requests == 0

    def test_tracked_agents(self):
        limiter = AgentRateLimiter()
        limiter.check("a1", "s1", ExecutionRing.RING_2_STANDARD)
        limiter.check("a2", "s1", ExecutionRing.RING_2_STANDARD)
        assert limiter.tracked_agents == 2

    def test_token_bucket_refill(self):
        bucket = TokenBucket(capacity=10, tokens=0, refill_rate=1000)
        # Refill should add tokens based on elapsed time
        import time
        time.sleep(0.01)
        assert bucket.available > 0


# ── Kill Switch Tests ───────────────────────────────────────────


class TestKillSwitch:
    def test_kill_with_handoff(self):
        ks = KillSwitch()
        ks.register_substitute("s1", "backup-agent")

        result = ks.kill(
            agent_did="bad-agent",
            session_id="s1",
            reason=KillReason.BEHAVIORAL_DRIFT,
            in_flight_steps=[
                {"step_id": "step-1", "saga_id": "saga-1"},
            ],
        )
        assert result.handoff_success_count == 1
        assert result.handoffs[0].to_agent == "backup-agent"
        assert result.handoffs[0].status == HandoffStatus.HANDED_OFF
        assert not result.compensation_triggered

    def test_kill_without_substitute(self):
        ks = KillSwitch()
        result = ks.kill(
            agent_did="bad-agent",
            session_id="s1",
            reason=KillReason.RATE_LIMIT,
            in_flight_steps=[
                {"step_id": "step-1", "saga_id": "saga-1"},
            ],
        )
        assert result.handoff_success_count == 0
        assert result.compensation_triggered

    def test_kill_no_in_flight_steps(self):
        ks = KillSwitch()
        result = ks.kill(
            agent_did="bad-agent",
            session_id="s1",
            reason=KillReason.MANUAL,
        )
        assert len(result.handoffs) == 0
        assert not result.compensation_triggered

    def test_killed_agent_removed_from_substitutes(self):
        ks = KillSwitch()
        ks.register_substitute("s1", "agent-a")
        ks.register_substitute("s1", "agent-b")
        ks.kill("agent-a", "s1", KillReason.RING_BREACH)
        # agent-a should no longer be a substitute
        result = ks.kill("agent-b", "s1", KillReason.MANUAL, [{"step_id": "s1", "saga_id": "sg1"}])
        # No substitutes left (agent-a was removed, agent-b is being killed)
        assert result.compensation_triggered

    def test_kill_history(self):
        ks = KillSwitch()
        ks.kill("a1", "s1", KillReason.MANUAL)
        ks.kill("a2", "s1", KillReason.RATE_LIMIT)
        assert ks.total_kills == 2

    def test_total_handoffs(self):
        ks = KillSwitch()
        ks.register_substitute("s1", "backup")
        ks.kill("a1", "s1", KillReason.MANUAL, [{"step_id": "s1", "saga_id": "sg1"}])
        assert ks.total_handoffs == 1

    def test_unregister_substitute(self):
        ks = KillSwitch()
        ks.register_substitute("s1", "backup")
        ks.unregister_substitute("s1", "backup")
        result = ks.kill("a1", "s1", KillReason.MANUAL, [{"step_id": "s1", "saga_id": "sg1"}])
        assert result.compensation_triggered
