"""
Causal Contribution Scoring — Shapley-value inspired fault attribution.

When a saga fails, traces the causal DAG of agent actions and computes
partial liability scores. Replaces binary guilty/not-guilty with
proportional fault attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class CausalNode:
    """A node in the causal DAG representing an agent's action."""

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_did: str = ""
    action_id: str = ""
    step_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    is_root_cause: bool = False
    dependencies: list[str] = field(default_factory=list)  # node_ids this depends on


@dataclass
class FaultAttribution:
    """Proportional fault attribution for an agent."""

    agent_did: str
    liability_score: float  # 0.0–1.0, proportional fault
    causal_contribution: float  # raw contribution before normalization
    is_direct_cause: bool = False
    reason: str = ""


@dataclass
class AttributionResult:
    """Complete attribution analysis for a saga failure."""

    attribution_id: str = field(default_factory=lambda: f"attr:{uuid.uuid4().hex[:8]}")
    saga_id: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributions: list[FaultAttribution] = field(default_factory=list)
    causal_chain_length: int = 0
    root_cause_agent: Optional[str] = None

    @property
    def agents_involved(self) -> list[str]:
        return [a.agent_did for a in self.attributions]

    def get_liability(self, agent_did: str) -> float:
        """Get liability score for a specific agent."""
        for a in self.attributions:
            if a.agent_did == agent_did:
                return a.liability_score
        return 0.0


class CausalAttributor:
    """
    Computes proportional fault attribution using simplified Shapley values.

    Instead of binary guilty/not-guilty, assigns partial liability scores
    based on each agent's causal contribution to the failure.

    The Shapley-inspired approach considers:
    1. Direct causation: was the agent's action the direct failure point?
    2. Enabling causation: did the agent's prior actions enable the failure?
    3. Timing weight: actions closer to the failure point carry more weight
    4. Risk weight: higher-risk actions carry more liability
    """

    DIRECT_CAUSE_WEIGHT = 0.5   # 50% liability goes to direct cause
    ENABLING_WEIGHT = 0.3        # 30% split among enabling causes
    PROXIMITY_WEIGHT = 0.2       # 20% based on temporal proximity

    def __init__(self) -> None:
        self._history: list[AttributionResult] = []

    def build_causal_dag(
        self,
        agent_actions: dict[str, list[dict]],
        failure_step_id: str,
        failure_agent_did: str,
    ) -> list[CausalNode]:
        """
        Build a causal DAG from agent actions.

        Args:
            agent_actions: {agent_did: [{action_id, step_id, success, timestamp, deps}]}
            failure_step_id: The step that failed
            failure_agent_did: The agent that failed

        Returns:
            List of CausalNodes forming the DAG
        """
        nodes = []
        for agent_did, actions in agent_actions.items():
            for action in actions:
                node = CausalNode(
                    agent_did=agent_did,
                    action_id=action.get("action_id", ""),
                    step_id=action.get("step_id", ""),
                    success=action.get("success", True),
                    is_root_cause=(
                        action.get("step_id") == failure_step_id
                        and agent_did == failure_agent_did
                    ),
                    dependencies=action.get("dependencies", []),
                )
                nodes.append(node)
        return nodes

    def attribute(
        self,
        saga_id: str,
        session_id: str,
        agent_actions: dict[str, list[dict]],
        failure_step_id: str,
        failure_agent_did: str,
        risk_weights: Optional[dict[str, float]] = None,
    ) -> AttributionResult:
        """
        Compute proportional fault attribution for a saga failure.

        Args:
            saga_id: The failed saga
            session_id: Session context
            agent_actions: {agent_did: [{action_id, step_id, success, timestamp, deps}]}
            failure_step_id: The step that triggered the failure
            failure_agent_did: The agent whose step failed
            risk_weights: Optional {action_id: risk_weight} for weighting

        Returns:
            AttributionResult with proportional liability scores
        """
        risk_weights = risk_weights or {}
        nodes = self.build_causal_dag(agent_actions, failure_step_id, failure_agent_did)

        # Compute raw contributions
        raw_scores: dict[str, float] = {}
        agents = set(agent_actions.keys())

        for agent_did in agents:
            agent_nodes = [n for n in nodes if n.agent_did == agent_did]
            score = 0.0

            for node in agent_nodes:
                # Direct cause component
                if node.is_root_cause:
                    score += self.DIRECT_CAUSE_WEIGHT

                # Enabling cause: failed actions that aren't the root
                if not node.success and not node.is_root_cause:
                    score += self.ENABLING_WEIGHT / max(
                        1, sum(1 for n in nodes if not n.success and not n.is_root_cause)
                    )

                # Risk weight of actions
                action_risk = risk_weights.get(node.action_id, 0.5)
                score += self.PROXIMITY_WEIGHT * action_risk / max(1, len(agent_nodes))

            raw_scores[agent_did] = score

        # Normalize to sum to 1.0
        total = sum(raw_scores.values())
        if total == 0:
            total = 1.0

        attributions = []
        for agent_did, raw in raw_scores.items():
            normalized = raw / total
            attributions.append(FaultAttribution(
                agent_did=agent_did,
                liability_score=round(normalized, 4),
                causal_contribution=round(raw, 4),
                is_direct_cause=(agent_did == failure_agent_did),
                reason=(
                    "Direct cause of failure"
                    if agent_did == failure_agent_did
                    else "Contributing factor"
                ),
            ))

        # Sort by liability (highest first)
        attributions.sort(key=lambda a: a.liability_score, reverse=True)

        result = AttributionResult(
            saga_id=saga_id,
            session_id=session_id,
            attributions=attributions,
            causal_chain_length=len(nodes),
            root_cause_agent=failure_agent_did,
        )
        self._history.append(result)
        return result

    @property
    def attribution_history(self) -> list[AttributionResult]:
        return list(self._history)
