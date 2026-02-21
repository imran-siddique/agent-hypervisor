<div align="center">

# Agent Hypervisor

**Runtime supervisor for multi-agent Shared Sessions with Execution Rings, Joint Liability, and Saga Orchestration**

[![GitHub Stars](https://img.shields.io/github/stars/imran-siddique/agent-hypervisor?style=social)](https://github.com/imran-siddique/agent-hypervisor/stargazers)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4%EF%B8%8F-ff69b4)](https://github.com/sponsors/imran-siddique)
[![CI](https://github.com/imran-siddique/agent-hypervisor/actions/workflows/ci.yml/badge.svg)](https://github.com/imran-siddique/agent-hypervisor/actions)
[![Tests](https://img.shields.io/badge/tests-326%20passing-brightgreen)](https://github.com/imran-siddique/agent-hypervisor)
[![Benchmark](https://img.shields.io/badge/latency-268%CE%BCs%20pipeline-orange)](benchmarks/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/agent-hypervisor/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Discussions](https://img.shields.io/github/discussions/imran-siddique/agent-hypervisor)](https://github.com/imran-siddique/agent-hypervisor/discussions)

> :star: **If this project helps you, please star it!** It helps others discover Agent Hypervisor.

> :link: **Part of the Agent Governance Ecosystem** -- Works with [Agent OS](https://github.com/imran-siddique/agent-os) (kernel), [AgentMesh](https://github.com/imran-siddique/agent-mesh) (trust network), and [Agent SRE](https://github.com/imran-siddique/agent-sre) (reliability)

> 📦 **Install the full stack:** `pip install ai-agent-governance[full]` — [PyPI](https://pypi.org/project/ai-agent-governance/) | [GitHub](https://github.com/imran-siddique/agent-governance)

[Quick Start](#quick-start) | [Why a Hypervisor?](#why-a-hypervisor) | [Features](#key-features) | [Performance](#performance) | [Modules](#modules) | [Ecosystem](#ecosystem)

</div>

### Integrated Into Major AI Frameworks

<p align="center">
  <a href="https://github.com/langgenius/dify-plugins/pull/2060"><img src="https://img.shields.io/badge/Dify-65K_%E2%AD%90_Merged-success?style=flat-square" alt="Dify"></a>
  <a href="https://github.com/run-llama/llama_index/pull/20644"><img src="https://img.shields.io/badge/LlamaIndex-47K_%E2%AD%90_Merged-success?style=flat-square" alt="LlamaIndex"></a>
  <a href="https://github.com/nicepkg/awesome-github-copilot/pull/26"><img src="https://img.shields.io/badge/Awesome_Copilot-21.6K_%E2%AD%90_Merged-success?style=flat-square" alt="Awesome Copilot"></a>
  <a href="https://github.com/microsoft/agent-lightning/pull/478"><img src="https://img.shields.io/badge/Agent--Lightning-15K_%E2%AD%90_Merged-success?style=flat-square" alt="Agent-Lightning"></a>
  <img src="https://img.shields.io/badge/Open_PRs-25+-blue?style=flat-square" alt="Open PRs">
  <img src="https://img.shields.io/badge/Framework_Issues-94+-blue?style=flat-square" alt="Issues">
</p>

## Quick Start

```bash
pip install agent-hypervisor
```

```python
from hypervisor import Hypervisor, SessionConfig, ConsistencyMode

hv = Hypervisor()
session = await hv.create_session(
    config=SessionConfig(enable_audit=True),
    creator_did="did:mesh:admin",
)
ring = await hv.join_session(session.sso.session_id, "did:mesh:agent-1", sigma_raw=0.85)
# → RING_2_STANDARD (trusted agent)
```

## Why a Hypervisor?

Just as OS hypervisors isolate virtual machines and enforce resource boundaries, the **Agent Hypervisor** isolates AI agent sessions and enforces **governance boundaries**:

| OS Hypervisor | Agent Hypervisor |
|---------------|-----------------|
| CPU rings (Ring 0–3) | **Execution Rings** — privilege levels based on trust score (σ_eff) |
| Process isolation | **Session isolation** — VFS namespacing, DID-bound identity |
| Memory protection | **Liability protection** — bonded reputation, collateral slashing |
| System calls | **Saga transactions** — multi-step operations with automatic rollback |
| Audit logs | **Merkle-chained delta audit** — tamper-evident forensic trail |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      AGENT HYPERVISOR                        │
│                                                              │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────────────┐ │
│  │   Session    │ │    Ring      │ │   Semantic Saga        │ │
│  │   Manager    │ │   Enforcer   │ │   Orchestrator         │ │
│  │             │ │              │ │  ┌──────────────────┐  │ │
│  │  SSO + VFS  │ │  Ring 0–3    │ │  │ Timeout + Retry  │  │ │
│  │  Lifecycle  │ │  σ_eff gates │ │  │ Compensation     │  │ │
│  └──────┬──────┘ └──────┬───────┘ │  │ Escalation       │  │ │
│         │               │         │  └──────────────────┘  │ │
│  ┌──────┴──────┐ ┌──────┴───────┐ └────────────┬───────────┘ │
│  │  Liability  │ │ Reversibility│               │            │
│  │  Engine     │ │  Registry    │ ┌─────────────┴──────────┐ │
│  │             │ │              │ │   Delta Audit Engine    │ │
│  │  Vouch +    │ │  Execute/    │ │                        │ │
│  │  Bond +     │ │  Undo API    │ │  Merkle Chain + GC     │ │
│  │  Slash      │ │  Mapping     │ │  Blockchain Commit     │ │
│  └─────────────┘ └──────────────┘ └────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
         │                │                    │
┌────────┴────────────────┴────────────────────┴───────────────┐
│                   AGENT-OS KERNEL LAYER                       │
│  ┌────────┐ ┌──────┐ ┌──────┐ ┌─────┐ ┌──────────┐          │
│  │ IATP   │ │ CMVK │ │Nexus │ │CaaS │ │  SCAK    │          │
│  └────────┘ └──────┘ └──────┘ └─────┘ └──────────┘          │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

### 🔐 Execution Rings (Hardware-Inspired Privilege Model)

```
Ring 0 (Root)       — Hypervisor config & slashing — requires SRE Witness
Ring 1 (Privileged) — Non-reversible actions — requires σ_eff > 0.95 + consensus
Ring 2 (Standard)   — Reversible actions — requires σ_eff > 0.60
Ring 3 (Sandbox)    — Read-only / research — default for unknown agents
```

Agents are automatically assigned to rings based on their effective trust score. Ring demotion happens in real-time if trust drops.

**v2.0:** Dynamic ring elevation (sudo with TTL), ring breach detection with circuit breakers, ring inheritance for spawned agents.

### 🤝 Joint Liability (Trust as Collateral)

High-trust agents can **vouch** for low-trust agents by bonding a percentage of their reputation:

```
σ_eff = σ_low + (ω × σ_high_bonded)
```

If the vouchee violates intent, **both agents are penalized** — the voucher's collateral is slashed. Max exposure limits (default: 80% of σ) prevent over-bonding.

**v2.0:** Shapley-value fault attribution (proportional blame, not binary), quarantine-before-terminate, persistent liability ledger for admission decisions.

### 🔄 Semantic Saga Orchestrator

Multi-step agent transactions with:
- **Timeout enforcement** — steps that hang are automatically cancelled
- **Retry with backoff** — transient failures retry with exponential delay
- **Reverse-order compensation** — on failure, all committed steps are undone
- **Escalation** — if compensation fails, Joint Liability slashing is triggered

**v2.0:** Parallel fan-out (ALL/MAJORITY/ANY policies), semantic checkpoints for partial replay, declarative YAML/dict DSL.

### 🔒 Session Consistency (NEW in v2.0)

- **Vector clocks** — causal consistency for shared VFS state
- **Intent locks** — READ/WRITE/EXCLUSIVE with deadlock detection
- **Isolation levels** — SNAPSHOT, READ_COMMITTED, SERIALIZABLE per saga

### 🛡️ Security (NEW in v2.0)

- **Rate limiting** — token bucket per agent per ring (sandbox: 5 rps, root: 100 rps)
- **Kill switch** — graceful termination with saga step handoff to substitute agents

### 📡 Observability (NEW in v2.0)

- **Structured event bus** — every hypervisor action emits typed events
- **Causal trace IDs** — distributed tracing with full delegation tree encoding

### 📋 Delta Audit Engine

Forensic-grade audit trails using:
- **Semantic diffs** — captures what changed, not full snapshots
- **Merkle chaining** — each delta references its parent hash (tamper-evident)
- **Blockchain commitment** — Summary Hash anchored on-chain at session end
- **Garbage collection** — ephemeral data purged, forensic artifacts retained

## Performance

| Operation | Mean Latency | Throughput |
|-----------|-------------|------------|
| Ring computation | **0.3μs** | 3.75M ops/s |
| Delta audit capture | **27μs** | 26K ops/s |
| Session lifecycle | **54μs** | 15.7K ops/s |
| 3-step saga | **151μs** | 5.3K ops/s |
| **Full governance pipeline** | **268μs** | **2,983 ops/s** |

> Full pipeline = session create + agent join + 3 audit deltas + saga step + terminate with Merkle root

## Installation

```bash
pip install agent-hypervisor
```

## Quick Start

```python
from hypervisor import Hypervisor, SessionConfig, ConsistencyMode

hv = Hypervisor()

# Create a shared session
session = await hv.create_session(
    config=SessionConfig(
        consistency_mode=ConsistencyMode.EVENTUAL,
        max_participants=5,
        min_sigma_eff=0.60,
    ),
    creator_did="did:mesh:admin",
)

# Agents join via IATP handshake — ring assigned by trust score
ring = await hv.join_session(
    session.sso.session_id,
    agent_did="did:mesh:agent-alpha",
    sigma_raw=0.85,
)
# → ExecutionRing.RING_2_STANDARD

# Activate and execute
await hv.activate_session(session.sso.session_id)

# Multi-step saga with automatic compensation
saga = session.saga.create_saga(session.sso.session_id)
step = session.saga.add_step(
    saga.saga_id, "draft_email", "did:mesh:agent-alpha",
    execute_api="/api/draft", undo_api="/api/undo-draft",
    timeout_seconds=30, max_retries=2,
)
result = await session.saga.execute_step(
    saga.saga_id, step.step_id, executor=draft_email
)

# Terminate — returns Merkle root Summary Hash
merkle_root = await hv.terminate_session(session.sso.session_id)
```

## Modules

| Module | Description | Tests |
|--------|-------------|-------|
| `hypervisor.session` | Shared Session Object lifecycle + VFS | 52 |
| `hypervisor.rings` | 4-ring privilege + elevation + breach detection | 34 |
| `hypervisor.liability` | Vouching, slashing, attribution, quarantine, ledger | 39 |
| `hypervisor.reversibility` | Execute/Undo API registry | 4 |
| `hypervisor.saga` | Saga orchestrator + fan-out + checkpoints + DSL | 41 |
| `hypervisor.audit` | Delta engine, Merkle chain, GC, commitment | 10 |
| `hypervisor.verification` | DID transaction history verification | 4 |
| `hypervisor.observability` | Event bus, causal trace IDs | 22 |
| `hypervisor.security` | Rate limiter, kill switch | 16 |
| `hypervisor.integrations` | Nexus, CMVK, IATP cross-module adapters | -- |
| **Integration** | End-to-end lifecycle, edge cases, security | **24** |
| **Scenarios** | Cross-module governance pipelines (7 suites) | **18** |
| **Total** | | **326** |

## Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run benchmarks
python benchmarks/bench_hypervisor.py
```

## Cross-Module Integrations

The Hypervisor integrates with other Agent-OS modules via adapters in `hypervisor.integrations`:

### Nexus Adapter — Trust-Scored Ring Assignment

```python
from hypervisor.integrations.nexus_adapter import NexusAdapter
from nexus.reputation import ReputationEngine

nexus = NexusAdapter(scorer=ReputationEngine())
sigma = nexus.resolve_sigma("did:mesh:agent-1", history=agent_history)
# → 0.82 (Nexus 820/1000 normalized)

ring = await hv.join_session(session_id, "did:mesh:agent-1", sigma_raw=sigma)
# → RING_2_STANDARD

# Report slashing back to Nexus for persistent reputation loss
nexus.report_slash("did:mesh:agent-1", reason="Behavioral drift", severity="high")
```

### CMVK Adapter — Behavioral Drift Detection

```python
from hypervisor.integrations.cmvk_adapter import CMVKAdapter

cmvk = CMVKAdapter(verifier=cmvk_engine)
result = cmvk.check_behavioral_drift(
    agent_did="did:mesh:agent-1",
    session_id=session_id,
    claimed_embedding=manifest_vector,
    observed_embedding=output_vector,
)

if result.should_slash:
    hv.slashing.slash(...)  # Trigger liability cascade
```

### IATP Adapter — Capability Manifest Parsing

```python
from hypervisor.integrations.iatp_adapter import IATPAdapter

iatp = IATPAdapter()
analysis = iatp.analyze_manifest(manifest)  # or analyze_manifest_dict(dict)
# → ManifestAnalysis with ring_hint, sigma_hint, actions, reversibility flags

ring = await hv.join_session(
    session_id, analysis.agent_did,
    actions=analysis.actions, sigma_raw=analysis.sigma_hint,
)
```

### REST API

Full FastAPI REST API with 22 endpoints and interactive Swagger docs:

```bash
pip install agent-hypervisor[api]
uvicorn hypervisor.api.server:app
# Open http://localhost:8000/docs for Swagger UI
```

Endpoints: Sessions, Rings, Sagas, Liability, Events, Health.

### Visualization Dashboard

Interactive Streamlit dashboard with 5 tabs:

```bash
cd examples/dashboard
pip install -r requirements.txt
streamlit run app.py
```

Tabs: Session Overview | Execution Rings | Saga Orchestration | Liability & Trust | Event Stream

## Ecosystem

Agent Hypervisor is part of the **Agent Governance Ecosystem** — four specialized repos that work together:

`
┌─────────────────────────────────────────────────────────────┐
│                  Agent Governance Ecosystem                   │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Agent OS    │  │  Agent Mesh  │  │    Agent SRE     │   │
│  │  Governance   │  │   Trust      │  │   Reliability    │   │
│  │  Kernel       │  │   Network    │  │   Platform       │   │
│  └──────┬───────┘  └──────┬───────┘  └───────┬──────────┘   │
│         │                 │                   │               │
│         └─────────┬───────┴───────────┬───────┘               │
│                   │                   │                       │
│          ┌────────┴───────────────────┴────────┐              │
│          │        Agent Hypervisor              │              │
│          │  Runtime supervisor for all agents   │              │
│          └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
`

| Repo | Role | Stars |
|------|------|-------|
| [Agent OS](https://github.com/imran-siddique/agent-os) | Policy enforcement kernel | 1,500+ tests |
| [Agent Mesh](https://github.com/imran-siddique/agent-mesh) | Cryptographic trust network | 1,400+ tests |
| [Agent SRE](https://github.com/imran-siddique/agent-sre) | SLO, chaos, cost guardrails | 1,070+ tests |
| **Agent Hypervisor** | Session isolation & governance runtime | 326 tests |

## Frequently Asked Questions

**Why use a hypervisor for AI agents?**
Just as OS hypervisors isolate virtual machines and enforce resource boundaries, an agent hypervisor isolates AI agent sessions and enforces governance boundaries. Without isolation, a misbehaving agent in a shared session can corrupt state, escalate privileges, or cascade failures across the entire system.

**How do Execution Rings differ from traditional access control?**
Traditional access control is static and binary (allowed/denied). Execution Rings are dynamic and graduated -- agents earn ring privileges based on their trust score, can request temporary elevation with TTL (like `sudo`), and are automatically demoted when trust drops. Ring breach detection catches anomalous behavior before damage occurs.

**What happens when a multi-agent saga fails?**
The Saga Orchestrator triggers reverse-order compensation for all committed steps. For parallel fan-out sagas, the failure policy determines the response: ALL_MUST_SUCCEED compensates if any branch fails, MAJORITY allows minority failures, and ANY succeeds if at least one branch completes. Semantic checkpoints enable partial replay without re-running completed effects.

**How does Shapley-value fault attribution work?**
When a saga fails, the hypervisor traces the causal DAG and assigns proportional blame: 50% weight to direct cause, 30% to enabling factors, 20% to temporal proximity. This prevents unfairly penalizing agents that merely contributed to but didn't directly cause a failure.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

- :bug: [Report a Bug](https://github.com/imran-siddique/agent-hypervisor/issues/new?labels=bug)
- :bulb: [Request a Feature](https://github.com/imran-siddique/agent-hypervisor/issues/new?labels=enhancement)
- :speech_balloon: [Join Discussions](https://github.com/imran-siddique/agent-hypervisor/discussions)
- Look for issues labeled [`good first issue`](https://github.com/imran-siddique/agent-hypervisor/labels/good%20first%20issue) to get started

## License

MIT -- see [LICENSE](LICENSE).

---

<div align="center">

**[Agent OS](https://github.com/imran-siddique/agent-os)** | **[AgentMesh](https://github.com/imran-siddique/agent-mesh)** | **[Agent SRE](https://github.com/imran-siddique/agent-sre)** | **[Agent Hypervisor](https://github.com/imran-siddique/agent-hypervisor)**

*Built with :heart: for the AI agent governance community*

If Agent Hypervisor helps your work, please consider giving it a :star:

</div>
