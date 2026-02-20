"""
Parallel Saga Fan-Out — concurrent step execution with failure policies.

Supports ALL_MUST_SUCCEED, MAJORITY_MUST_SUCCEED, and ANY_MUST_SUCCEED
semantics per fan-out branch, with automatic compensation routing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import uuid

from hypervisor.saga.state_machine import SagaStep, StepState


class FanOutPolicy(str, Enum):
    """Failure policy for parallel fan-out branches."""

    ALL_MUST_SUCCEED = "all_must_succeed"
    MAJORITY_MUST_SUCCEED = "majority_must_succeed"
    ANY_MUST_SUCCEED = "any_must_succeed"


@dataclass
class FanOutBranch:
    """A single branch in a fan-out group."""

    branch_id: str = field(default_factory=lambda: f"branch:{uuid.uuid4().hex[:8]}")
    step: Optional[SagaStep] = None
    result: Any = None
    error: Optional[str] = None
    succeeded: bool = False


@dataclass
class FanOutGroup:
    """A group of parallel branches with a failure policy."""

    group_id: str = field(default_factory=lambda: f"fanout:{uuid.uuid4().hex[:8]}")
    saga_id: str = ""
    policy: FanOutPolicy = FanOutPolicy.ALL_MUST_SUCCEED
    branches: list[FanOutBranch] = field(default_factory=list)
    resolved: bool = False
    policy_satisfied: bool = False
    compensation_needed: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for b in self.branches if b.succeeded)

    @property
    def failure_count(self) -> int:
        return sum(1 for b in self.branches if not b.succeeded and b.error)

    @property
    def total_branches(self) -> int:
        return len(self.branches)

    def check_policy(self) -> bool:
        """Check if the failure policy is satisfied."""
        if self.policy == FanOutPolicy.ALL_MUST_SUCCEED:
            return self.success_count == self.total_branches
        elif self.policy == FanOutPolicy.MAJORITY_MUST_SUCCEED:
            return self.success_count > self.total_branches / 2
        elif self.policy == FanOutPolicy.ANY_MUST_SUCCEED:
            return self.success_count >= 1
        return False


class FanOutOrchestrator:
    """
    Orchestrates parallel saga fan-out with configurable failure policies.

    Usage:
        fan = FanOutOrchestrator()
        group = fan.create_group(saga_id, FanOutPolicy.MAJORITY_MUST_SUCCEED)
        fan.add_branch(group.group_id, step1)
        fan.add_branch(group.group_id, step2)
        fan.add_branch(group.group_id, step3)
        result = await fan.execute(group.group_id, executor_map)
    """

    def __init__(self) -> None:
        self._groups: dict[str, FanOutGroup] = {}

    def create_group(
        self,
        saga_id: str,
        policy: FanOutPolicy = FanOutPolicy.ALL_MUST_SUCCEED,
    ) -> FanOutGroup:
        """Create a new fan-out group."""
        group = FanOutGroup(saga_id=saga_id, policy=policy)
        self._groups[group.group_id] = group
        return group

    def add_branch(
        self,
        group_id: str,
        step: SagaStep,
    ) -> FanOutBranch:
        """Add a branch to a fan-out group."""
        group = self._get_group(group_id)
        branch = FanOutBranch(step=step)
        group.branches.append(branch)
        return branch

    async def execute(
        self,
        group_id: str,
        executors: dict[str, Callable[..., Any]],
        timeout_seconds: int = 300,
    ) -> FanOutGroup:
        """
        Execute all branches in parallel.

        Args:
            group_id: Fan-out group to execute
            executors: Map of step_id -> async callable
            timeout_seconds: Overall timeout for the fan-out

        Returns:
            FanOutGroup with results
        """
        group = self._get_group(group_id)

        async def run_branch(branch: FanOutBranch) -> None:
            if not branch.step:
                branch.error = "No step assigned"
                return

            executor = executors.get(branch.step.step_id)
            if not executor:
                branch.error = f"No executor for step {branch.step.step_id}"
                return

            try:
                branch.step.transition(StepState.EXECUTING)
                result = await asyncio.wait_for(
                    executor(),
                    timeout=branch.step.timeout_seconds,
                )
                branch.result = result
                branch.succeeded = True
                branch.step.execute_result = result
                branch.step.transition(StepState.COMMITTED)
            except Exception as e:
                branch.error = str(e)
                branch.succeeded = False
                branch.step.error = str(e)
                branch.step.transition(StepState.FAILED)

        # Execute all branches concurrently
        tasks = [run_branch(branch) for branch in group.branches]
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout_seconds,
        )

        # Check policy
        group.policy_satisfied = group.check_policy()
        group.resolved = True

        # Determine which branches need compensation
        if not group.policy_satisfied:
            # All succeeded branches need compensation
            group.compensation_needed = [
                b.step.step_id
                for b in group.branches
                if b.succeeded and b.step
            ]
        elif group.policy != FanOutPolicy.ALL_MUST_SUCCEED:
            # Policy satisfied but some failed — no compensation needed
            # (the policy allows partial failure)
            pass

        return group

    def get_group(self, group_id: str) -> Optional[FanOutGroup]:
        return self._groups.get(group_id)

    def _get_group(self, group_id: str) -> FanOutGroup:
        group = self._groups.get(group_id)
        if not group:
            raise ValueError(f"Fan-out group {group_id} not found")
        return group

    @property
    def active_groups(self) -> list[FanOutGroup]:
        return [g for g in self._groups.values() if not g.resolved]
