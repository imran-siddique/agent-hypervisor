# Tutorial: Build a Supervised Multi-Agent Workflow with Execution Rings

This tutorial demonstrates how the **agent-hypervisor** execution ring system
enforces privilege isolation across agents in a supervised multi-agent workflow.

## What you'll learn

1. Assign agents to execution rings with different privilege levels
2. Define actions with ring requirements and enforce access control
3. Detect and deny unauthorized cross-ring access attempts
4. Temporarily elevate an agent's ring for emergency operations
5. Kill a misbehaving agent with the supervisor kill switch
6. Inspect the full audit trail of ring transitions

## Concepts

### Execution rings

Execution rings are inspired by hardware CPU protection rings. Each agent runs
at a specific ring level that determines what actions it can perform. **Lower
numbers = more privilege.**

```
┌──────────────────────────────────────────────────┐
│  Ring 0 — Root / Supervisor (kernel)             │
│  Full control: config, penalties, kill switch    │
│  ┌──────────────────────────────────────────┐    │
│  │  Ring 1 — Privileged                     │    │
│  │  Non-reversible actions: DB writes,      │    │
│  │  deployments. Requires eff_score > 0.95  │    │
│  │  ┌──────────────────────────────────┐    │    │
│  │  │  Ring 2 — Standard               │    │    │
│  │  │  Reversible actions: read data,  │    │    │
│  │  │  compute, draft reports.         │    │    │
│  │  │  Requires eff_score > 0.60       │    │    │
│  │  │  ┌──────────────────────────┐    │    │    │
│  │  │  │  Ring 3 — Sandbox        │    │    │    │
│  │  │  │  Read-only / research.   │    │    │    │
│  │  │  │  Default for untrusted   │    │    │    │
│  │  │  │  agents.                 │    │    │    │
│  │  │  └──────────────────────────┘    │    │    │
│  │  └──────────────────────────────────┘    │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

| Ring | Name | Who | Can do |
|------|------|-----|--------|
| 0 | Root | Supervisor / hypervisor | Everything — config, penalties, kill switch |
| 1 | Privileged | Trusted data agents | Non-reversible actions (DB writes, deployments) |
| 2 | Standard | Analysis agents | Reversible actions (compute, draft, read/write) |
| 3 | Sandbox | User-facing / untrusted | Read-only — can only return pre-approved responses |

### Why ring isolation matters

Without ring isolation, any agent can perform any action — a user-facing
chatbot could accidentally (or maliciously) delete a database. Rings enforce
**least-privilege by default:**

- A sandboxed agent **cannot** write to the database even if it tries
- A standard agent **cannot** execute admin operations
- Privilege escalation requires **explicit elevation** with a time-limited grant
- The supervisor can **kill** any agent that violates its ring boundaries

### Key classes

| Class | Module | Role |
|---|---|---|
| `ExecutionRing` | `hypervisor.models` | Enum defining ring levels (0–3) |
| `ActionDescriptor` | `hypervisor.models` | Action with ring requirements derived from reversibility |
| `RingEnforcer` | `hypervisor.rings.enforcer` | Checks if an agent's ring permits an action |
| `RingElevationManager` | `hypervisor.rings.elevation` | Manages temporary privilege escalation |
| `ActionClassifier` | `hypervisor.rings.classifier` | Classifies actions into ring levels and risk weights |
| `KillSwitch` | `hypervisor.security.kill_switch` | Terminates misbehaving agents |
| `RingBreachDetector` | `hypervisor.rings.breach_detector` | Detects anomalous cross-ring access patterns |

## The scenario

We model a **data processing pipeline** with four agents at different ring levels:

| Agent | Ring | Role | Capabilities |
|-------|------|------|-------------|
| `supervisor` | 0 (Root) | Orchestrator | Full control — manages workflow, kills agents |
| `data-agent` | 1 (Privileged) | Data engineer | Read databases, write reports, run ETL |
| `analysis-agent` | 2 (Standard) | Analyst | Read data, compute aggregations, draft results |
| `user-agent` | 3 (Sandbox) | User-facing | Return pre-approved responses only |

The demo walks through five scenarios:

1. **Normal operation** — each agent performs actions within its ring
2. **Ring enforcement** — `user-agent` (Ring 3) tries to write to the database → DENIED
3. **Ring elevation** — `analysis-agent` (Ring 2) gets temporary sudo to Ring 1
4. **Kill switch** — supervisor kills `user-agent` for attempting a ring breach
5. **Audit trail** — review all ring transitions, denials, and kills

## Running the demo

```bash
# From the repository root
cd tutorials/execution-rings-workflow
python demo.py
```

No API keys or external services required — all operations are simulated
locally with mock functions.

## Expected output

```
══════════════════════════════════════════════════════════════
  Execution Rings Workflow Demo — Data Processing Pipeline
══════════════════════════════════════════════════════════════

── Agent Roster ──────────────────────────────────────────────
  supervisor         Ring 0 (root)          Orchestrator
  data-agent         Ring 1 (privileged)    Data Engineer
  analysis-agent     Ring 2 (standard)      Analyst
  user-agent         Ring 3 (sandbox)       User-facing

━━ Scenario 1: Normal Operation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ data-agent       → write_report           ALLOWED (ring 1 ≤ required 1)
  ✓ analysis-agent   → read_dataset           ALLOWED (ring 2 ≤ required 3)
  ✓ analysis-agent   → compute_aggregation    ALLOWED (ring 2 ≤ required 2)
  ✓ user-agent       → return_response        ALLOWED (ring 3 ≤ required 3)

━━ Scenario 2: Ring Enforcement ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✗ user-agent       → write_report           DENIED  (ring 3 > required 1)
  ✗ user-agent       → compute_aggregation    DENIED  (ring 3 > required 2)
  ✗ analysis-agent   → write_report           DENIED  (ring 2 > required 1)

━━ Scenario 3: Ring Elevation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↑ analysis-agent   elevated Ring 2 → Ring 1 (reason: emergency data export)
  ✓ analysis-agent   → write_report           ALLOWED (elevated ring 1 ≤ required 1)
  ↓ analysis-agent   elevation revoked — back to Ring 2

━━ Scenario 4: Kill Switch ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠ user-agent       attempted ring breach — triggering kill switch
  ☠ user-agent       KILLED (reason: ring_breach)
    Kill ID: kill:xxxxxxxx
    Compensation triggered: False

━━ Scenario 5: Audit Trail ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  #  Event                Agent              Details
  1  ring_assigned        supervisor         → Ring 0
  2  ring_assigned        data-agent         → Ring 1
  3  ring_assigned        analysis-agent     → Ring 2
  4  ring_assigned        user-agent         → Ring 3
  5  access_granted       data-agent         write_report
  6  access_granted       analysis-agent     read_dataset
  7  access_granted       analysis-agent     compute_aggregation
  8  access_granted       user-agent         return_response
  9  access_denied        user-agent         write_report
  10 access_denied        user-agent         compute_aggregation
  11 access_denied        analysis-agent     write_report
  12 ring_elevated        analysis-agent     Ring 2 → Ring 1
  13 access_granted       analysis-agent     write_report (elevated)
  14 ring_revoked         analysis-agent     Ring 1 → Ring 2
  15 kill_executed        user-agent         ring_breach
══════════════════════════════════════════════════════════════
```

## How it works

### 1. Define actions with ring requirements

```python
from hypervisor.models import ActionDescriptor, ReversibilityLevel

# Non-reversible write → requires Ring 1 (privileged)
write_report = ActionDescriptor(
    action_id="write_report",
    name="Write Report to DB",
    execute_api="/reports/write",
    reversibility=ReversibilityLevel.NONE,
)

# Read-only action → requires Ring 3 (sandbox-safe)
read_dataset = ActionDescriptor(
    action_id="read_dataset",
    name="Read Dataset",
    execute_api="/data/read",
    is_read_only=True,
)
```

### 2. Enforce ring access

```python
from hypervisor.models import ExecutionRing
from hypervisor.rings.enforcer import RingEnforcer

enforcer = RingEnforcer()
result = enforcer.check(
    agent_ring=ExecutionRing.RING_3_SANDBOX,
    action=write_report,
    eff_score=0.5,
)
print(result.allowed)  # False — Ring 3 cannot perform Ring 1 actions
print(result.reason)   # "Agent ring 3 insufficient for required ring 1"
```

### 3. Elevate privileges temporarily

```python
from hypervisor.rings.elevation import RingElevationManager

elevator = RingElevationManager()
elevation = elevator.request_elevation(
    agent_did="analysis-agent",
    session_id="session-1",
    current_ring=ExecutionRing.RING_2_STANDARD,
    target_ring=ExecutionRing.RING_1_PRIVILEGED,
    ttl_seconds=60,
    reason="emergency data export",
)
```

> **Note:** Ring elevation is denied in the community edition. The demo
> simulates elevation logic to illustrate the concept.

### 4. Kill a misbehaving agent

```python
from hypervisor.security.kill_switch import KillSwitch, KillReason

kill_switch = KillSwitch()
result = kill_switch.kill(
    agent_did="user-agent",
    session_id="session-1",
    reason=KillReason.RING_BREACH,
    details="Attempted to write to database from Ring 3",
)
print(result.kill_id)  # "kill:a1b2c3d4"
```

## Next steps

- Integrate with the full `Hypervisor` orchestrator for session-managed rings
- Use `RingBreachDetector` to automatically detect anomalous access patterns
- Combine with `SagaOrchestrator` for transactional multi-step workflows
- Add `LiabilityMatrix` to track agent trust scores alongside ring assignments
- Explore the `ActionClassifier` for automatic ring-level classification
