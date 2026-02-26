"""
Agent Hypervisor v2.0

Runtime supervisor for multi-agent Shared Sessions with execution rings,
liability tracking, saga orchestration, and audit trails.

Usage:
    >>> from hypervisor import Hypervisor, SessionConfig, ConsistencyMode
    >>> hv = Hypervisor()
    >>> session = await hv.create_session(
    ...     config=SessionConfig(consistency_mode=ConsistencyMode.EVENTUAL)
    ... )

Version: 2.0.0
"""

__version__ = "2.0.0"

# Core models
from hypervisor.models import (
    ConsistencyMode,
    ExecutionRing,
    ReversibilityLevel,
    SessionConfig,
    SessionState,
)

# Session management
from hypervisor.session import SharedSessionObject
from hypervisor.session.sso import SessionVFS, VFSEdit, VFSPermissionError
from hypervisor.session.vector_clock import VectorClock, VectorClockManager, CausalViolationError
from hypervisor.session.intent_locks import IntentLockManager, LockIntent, LockContentionError, DeadlockError
from hypervisor.session.isolation import IsolationLevel

# Liability engine
from hypervisor.liability.vouching import VouchRecord, VouchingEngine
from hypervisor.liability.slashing import SlashingEngine
from hypervisor.liability import LiabilityMatrix
from hypervisor.liability.attribution import CausalAttributor, AttributionResult
from hypervisor.liability.quarantine import QuarantineManager, QuarantineReason
from hypervisor.liability.ledger import LiabilityLedger, LedgerEntryType

# Execution rings
from hypervisor.rings.enforcer import RingEnforcer
from hypervisor.rings.classifier import ActionClassifier
from hypervisor.rings.elevation import RingElevationManager, RingElevation, ElevationDenialReason
from hypervisor.rings.breach_detector import RingBreachDetector, BreachSeverity

# Reversibility
from hypervisor.reversibility.registry import ReversibilityRegistry

# Saga
from hypervisor.saga.orchestrator import SagaOrchestrator, SagaTimeoutError
from hypervisor.saga.state_machine import SagaState, StepState
from hypervisor.saga.fan_out import FanOutOrchestrator, FanOutPolicy
from hypervisor.saga.checkpoint import CheckpointManager, SemanticCheckpoint
from hypervisor.saga.dsl import SagaDSLParser, SagaDefinition

# Audit
from hypervisor.audit.delta import DeltaEngine
from hypervisor.audit.commitment import CommitmentEngine
from hypervisor.audit.gc import EphemeralGC

# Verification
from hypervisor.verification.history import TransactionHistoryVerifier

# Observability
from hypervisor.observability.event_bus import HypervisorEventBus, EventType, HypervisorEvent
from hypervisor.observability.causal_trace import CausalTraceId

# Security
from hypervisor.security.rate_limiter import AgentRateLimiter, RateLimitExceeded
from hypervisor.security.kill_switch import KillSwitch, KillResult

# Top-level orchestrator
from hypervisor.core import Hypervisor

__all__ = [
    # Version
    "__version__",
    # Core
    "Hypervisor",
    # Models
    "ConsistencyMode",
    "ExecutionRing",
    "ReversibilityLevel",
    "SessionConfig",
    "SessionState",
    # Session
    "SharedSessionObject",
    "SessionVFS",
    "VFSEdit",
    "VFSPermissionError",
    "VectorClock",
    "VectorClockManager",
    "CausalViolationError",
    "IntentLockManager",
    "LockIntent",
    "LockContentionError",
    "DeadlockError",
    "IsolationLevel",
    # Liability
    "VouchRecord",
    "VouchingEngine",
    "SlashingEngine",
    "LiabilityMatrix",
    "CausalAttributor",
    "AttributionResult",
    "QuarantineManager",
    "QuarantineReason",
    "LiabilityLedger",
    "LedgerEntryType",
    # Rings
    "RingEnforcer",
    "ActionClassifier",
    "RingElevationManager",
    "RingElevation",
    "ElevationDenialReason",
    "RingBreachDetector",
    "BreachSeverity",
    # Reversibility
    "ReversibilityRegistry",
    # Saga
    "SagaOrchestrator",
    "SagaTimeoutError",
    "SagaState",
    "StepState",
    "FanOutOrchestrator",
    "FanOutPolicy",
    "CheckpointManager",
    "SemanticCheckpoint",
    "SagaDSLParser",
    "SagaDefinition",
    # Audit
    "DeltaEngine",
    "CommitmentEngine",
    "EphemeralGC",
    # Verification
    "TransactionHistoryVerifier",
    # Observability
    "HypervisorEventBus",
    "EventType",
    "HypervisorEvent",
    "CausalTraceId",
    # Security
    "AgentRateLimiter",
    "RateLimitExceeded",
    "KillSwitch",
    "KillResult",
]
