"""
Declarative Saga DSL — define saga topology via dict/YAML.

Makes sagas auditable and testable independently of agent code.
Supports fan-out, compensation chains, and liability policies.

Example definition:
    saga_def = {
        "name": "deploy-model",
        "session_id": "sess-123",
        "steps": [
            {
                "id": "validate",
                "action_id": "model.validate",
                "agent": "did:mesh:validator",
                "execute_api": "/api/validate",
                "undo_api": "/api/rollback-validate",
            },
            {
                "id": "deploy",
                "action_id": "model.deploy",
                "agent": "did:mesh:deployer",
                "execute_api": "/api/deploy",
                "undo_api": "/api/rollback-deploy",
                "timeout": 600,
                "retries": 2,
            },
        ],
        "fan_out": [
            {
                "policy": "majority_must_succeed",
                "branches": ["test-a", "test-b", "test-c"],
            }
        ],
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import uuid

from hypervisor.saga.fan_out import FanOutPolicy
from hypervisor.saga.state_machine import SagaStep


@dataclass
class SagaDSLStep:
    """A step parsed from the DSL definition."""

    id: str = ""
    action_id: str = ""
    agent: str = ""
    execute_api: str = ""
    undo_api: Optional[str] = None
    timeout: int = 300
    retries: int = 0
    checkpoint_goal: Optional[str] = None


@dataclass
class SagaDSLFanOut:
    """A fan-out group parsed from the DSL definition."""

    policy: FanOutPolicy = FanOutPolicy.ALL_MUST_SUCCEED
    branch_step_ids: list[str] = field(default_factory=list)


@dataclass
class SagaDefinition:
    """A complete saga definition parsed from DSL."""

    name: str = ""
    session_id: str = ""
    saga_id: str = field(default_factory=lambda: f"saga:{uuid.uuid4().hex[:8]}")
    steps: list[SagaDSLStep] = field(default_factory=list)
    fan_outs: list[SagaDSLFanOut] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_ids(self) -> list[str]:
        return [s.id for s in self.steps]

    @property
    def fan_out_step_ids(self) -> set[str]:
        ids: set[str] = set()
        for fo in self.fan_outs:
            ids.update(fo.branch_step_ids)
        return ids

    @property
    def sequential_steps(self) -> list[SagaDSLStep]:
        """Steps not part of any fan-out group."""
        fo_ids = self.fan_out_step_ids
        return [s for s in self.steps if s.id not in fo_ids]


class SagaDSLParser:
    """
    Parses saga definitions from dict (or YAML-loaded dict).

    Validates structure and produces a SagaDefinition that can be
    used with SagaOrchestrator and FanOutOrchestrator.
    """

    def parse(self, definition: dict[str, Any]) -> SagaDefinition:
        """
        Parse a saga definition dict into a SagaDefinition.

        Raises:
            SagaDSLError: If the definition is invalid
        """
        name = definition.get("name", "")
        if not name:
            raise SagaDSLError("Saga definition must have a 'name'")

        session_id = definition.get("session_id", "")
        if not session_id:
            raise SagaDSLError("Saga definition must have a 'session_id'")

        # Parse steps
        raw_steps = definition.get("steps", [])
        if not raw_steps:
            raise SagaDSLError("Saga must have at least one step")

        steps = []
        step_ids = set()
        for raw in raw_steps:
            step = self._parse_step(raw)
            if step.id in step_ids:
                raise SagaDSLError(f"Duplicate step ID: {step.id}")
            step_ids.add(step.id)
            steps.append(step)

        # Parse fan-outs
        fan_outs = []
        for raw_fo in definition.get("fan_out", []):
            fo = self._parse_fan_out(raw_fo, step_ids)
            fan_outs.append(fo)

        return SagaDefinition(
            name=name,
            session_id=session_id,
            saga_id=definition.get("saga_id", f"saga:{uuid.uuid4().hex[:8]}"),
            steps=steps,
            fan_outs=fan_outs,
            metadata=definition.get("metadata", {}),
        )

    def _parse_step(self, raw: dict) -> SagaDSLStep:
        step_id = raw.get("id", "")
        if not step_id:
            raise SagaDSLError("Each step must have an 'id'")

        action_id = raw.get("action_id", "")
        if not action_id:
            raise SagaDSLError(f"Step {step_id} must have an 'action_id'")

        agent = raw.get("agent", "")
        if not agent:
            raise SagaDSLError(f"Step {step_id} must have an 'agent'")

        return SagaDSLStep(
            id=step_id,
            action_id=action_id,
            agent=agent,
            execute_api=raw.get("execute_api", ""),
            undo_api=raw.get("undo_api"),
            timeout=raw.get("timeout", 300),
            retries=raw.get("retries", 0),
            checkpoint_goal=raw.get("checkpoint_goal"),
        )

    def _parse_fan_out(self, raw: dict, valid_step_ids: set[str]) -> SagaDSLFanOut:
        policy_str = raw.get("policy", "all_must_succeed")
        try:
            policy = FanOutPolicy(policy_str)
        except ValueError:
            raise SagaDSLError(
                f"Invalid fan-out policy: {policy_str}. "
                f"Valid: {[p.value for p in FanOutPolicy]}"
            )

        branches = raw.get("branches", [])
        if len(branches) < 2:
            raise SagaDSLError("Fan-out must have at least 2 branches")

        for bid in branches:
            if bid not in valid_step_ids:
                raise SagaDSLError(f"Fan-out branch '{bid}' is not a valid step ID")

        return SagaDSLFanOut(policy=policy, branch_step_ids=branches)

    def to_saga_steps(self, definition: SagaDefinition) -> list[SagaStep]:
        """Convert a SagaDefinition into SagaStep objects."""
        return [
            SagaStep(
                step_id=s.id,
                action_id=s.action_id,
                agent_did=s.agent,
                execute_api=s.execute_api,
                undo_api=s.undo_api,
                timeout_seconds=s.timeout,
                max_retries=s.retries,
            )
            for s in definition.steps
        ]

    def validate(self, definition: dict[str, Any]) -> list[str]:
        """
        Validate a definition and return list of errors (empty = valid).
        """
        errors = []
        if not definition.get("name"):
            errors.append("Missing 'name'")
        if not definition.get("session_id"):
            errors.append("Missing 'session_id'")
        if not definition.get("steps"):
            errors.append("Missing 'steps'")
        else:
            step_ids = set()
            for i, step in enumerate(definition["steps"]):
                if not step.get("id"):
                    errors.append(f"Step {i} missing 'id'")
                elif step["id"] in step_ids:
                    errors.append(f"Duplicate step ID: {step['id']}")
                else:
                    step_ids.add(step["id"])
                if not step.get("action_id"):
                    errors.append(f"Step {step.get('id', i)} missing 'action_id'")
                if not step.get("agent"):
                    errors.append(f"Step {step.get('id', i)} missing 'agent'")
        return errors


class SagaDSLError(Exception):
    """Raised for invalid saga DSL definitions."""
