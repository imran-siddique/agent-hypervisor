# Contributing to Agent Hypervisor

Thank you for your interest in contributing to Agent Hypervisor! This project provides runtime isolation, execution rings, and governance for autonomous AI agents. This guide will help you get set up and make your first contribution.

---

## Table of Contents

- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Code Style](#-code-style)
- [Making Changes](#-making-changes)
- [Testing](#-testing)
- [Issue Guidelines](#-issue-guidelines)
- [Architecture Overview](#-architecture-overview)
- [Design Philosophy](#-design-philosophy)
- [Getting Help](#-getting-help)

---

## 🚀 Getting Started

### Fork and Clone

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/agent-hypervisor.git
cd agent-hypervisor

# 2. Add upstream remote
git remote add upstream https://github.com/imran-siddique/agent-hypervisor.git

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Run tests to verify your setup
python -m pytest

# 5. Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality on every commit. The hooks include:

- **Trailing whitespace** and **end-of-file** fixes
- **YAML validation** and **merge conflict** detection
- **Private key detection** (security)
- **Ruff** linting and formatting
- **mypy** type checking (excludes tests)
- **pytest** check on push

Hooks are configured in `.pre-commit-config.yaml`. After installing with `pre-commit install`, they run automatically on `git commit`.

---

## 🛠️ Development Setup

### Python Version

Agent Hypervisor requires **Python 3.11 or higher** and supports Python 3.11, 3.12, and 3.13. We recommend using [pyenv](https://github.com/pyenv/pyenv) to manage Python versions:

```bash
pyenv install 3.12
pyenv local 3.12
```

### Virtual Environment

Always use a virtual environment for development:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install all dev dependencies
pip install -e ".[dev]"
```

### Optional Extras

Depending on what you're working on, install additional extras:

```bash
pip install -e ".[dev,api]"       # FastAPI REST API server
pip install -e ".[dev,nexus]"     # Structured logging (structlog)
pip install -e ".[dev,full]"      # Everything
```

### Running the API Server

The optional API layer is powered by FastAPI and Uvicorn:

```bash
pip install -e ".[api]"
uvicorn hypervisor.api.server:app
```

By default the server listens on `http://127.0.0.1:8000`. Add `--reload` during development for auto-restart on file changes.

### IDE Recommendations

- **VS Code** with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff), and [mypy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) extensions
- **PyCharm** Professional with built-in type checking enabled
- Enable format-on-save with Ruff for consistent formatting

---

## 📝 Code Style

### Formatting and Linting

We use **Ruff** for linting and formatting, and **mypy** for type checking:

```bash
# Format code
ruff format src/ tests/

# Lint (with auto-fix)
ruff check src/ tests/ --fix

# Type check
mypy src/
```

### Rules

| Rule | Details |
|------|---------|
| **Line length** | 100 characters maximum |
| **Target version** | Python 3.11 |
| **Type hints** | Required on all function signatures |
| **Docstrings** | Required for all public modules, classes, and functions |
| **Docstring style** | Google style |
| **Import order** | Enforced by Ruff (`I` rule) — stdlib → third-party → local |
| **Lint rules** | `E`, `F`, `I`, `W`, `B`, `C4`, `UP` |

### Type Hints

All function signatures must include type hints. Use `from __future__ import annotations` for modern syntax:

```python
from __future__ import annotations

from hypervisor.rings.engine import ExecutionRing

def assign_ring(
    agent_did: str,
    sigma_raw: float,
    *,
    session_id: str | None = None,
) -> ExecutionRing:
    """Assign an execution ring based on agent trust score.

    Args:
        agent_did: The DID identifier of the agent.
        sigma_raw: The raw trust score (0.0–1.0).
        session_id: Optional session to bind the assignment to.

    Returns:
        The execution ring assigned to the agent.

    Raises:
        ValueError: If sigma_raw is outside the valid range.
    """
    ...
```

### Docstrings (Google Style)

```python
class SagaOrchestrator:
    """Orchestrates multi-step transactions with compensation.

    Manages saga lifecycle including step execution, timeout
    enforcement, retry with backoff, and reverse-order compensation
    on failure.

    Attributes:
        session_id: The session this saga belongs to.
        consistency_mode: Isolation level for saga execution.

    Example:
        >>> saga = orchestrator.create_saga(session_id="sess-1")
        >>> step = orchestrator.add_step(
        ...     saga.saga_id, "draft_email", "did:mesh:agent-1",
        ...     execute_api="/api/draft", undo_api="/api/undo-draft",
        ... )
        >>> result = await orchestrator.execute_step(saga.saga_id, step.step_id)
    """
```

---

## 🔀 Making Changes

### Branch Naming

Create a branch from `main` using one of these prefixes:

| Prefix | Use For |
|--------|---------|
| `feat/` | New features (e.g., `feat/session-timeout`) |
| `fix/` | Bug fixes (e.g., `fix/ring-escalation-bug`) |
| `docs/` | Documentation changes (e.g., `docs/api-examples`) |
| `test/` | Test additions/improvements (e.g., `test/kill-switch-edge-cases`) |
| `refactor/` | Code refactoring (e.g., `refactor/saga-engine`) |
| `chore/` | Maintenance tasks (e.g., `chore/update-deps`) |

```bash
# Sync with upstream before branching
git fetch upstream
git checkout -b feat/my-feature upstream/main
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `perf`

**Scopes** (optional): `session`, `rings`, `saga`, `liability`, `audit`, `security`, `api`, `observability`

**Examples:**

```
feat(saga): add parallel execution with fan-out policies
fix(rings): correct ring privilege check for Ring-2
docs: update README with new install steps
test(liability): add slash cascade edge cases
perf(audit): optimize hash-chain delta computation
```

### Pull Request Process

1. **Fork** the repository and create your branch
2. **Make changes** following the code style guidelines
3. **Write/update tests** — all new features need test coverage
4. **Run the full check suite:**
   ```bash
   ruff format src/ tests/
   ruff check src/ tests/
   mypy src/
   python -m pytest
   ```
5. **Push** your branch and open a Pull Request against `main`
6. **Fill out the PR description** with:
   - What changed and why
   - How to test the changes
   - Which module is affected
   - Related issue numbers (e.g., `Closes #11`)
7. **Address review feedback** — maintainers may request changes
8. **Merge** — a maintainer will merge once approved

### Review Criteria

PRs are evaluated on:
- Correctness and security implications
- Test coverage for new/changed behavior
- Type safety (mypy must pass with `--strict`)
- Documentation for public APIs
- Module independence (modules should work standalone)

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run a specific test file
python -m pytest tests/test_kill_switch.py -v

# Run a specific test
python -m pytest tests/test_breach_detector.py::test_breach_triggers_demotion -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# Run with coverage report
python -m pytest --cov=src/hypervisor --cov-report=html --cov-report=term-missing

# Run benchmarks
python benchmarks/bench_hypervisor.py
```

### Writing Tests

- **Test file location:** Place tests in `tests/` at the repo root; use `tests/unit/` for unit tests and `tests/integration/` for integration tests
- **Naming convention:** `test_<module>.py` for files, `test_<behavior>` for functions
- **Async tests:** Use `pytest-asyncio` — tests are auto-detected (`asyncio_mode = "auto"`)
- **Property-based tests:** Use [Hypothesis](https://hypothesis.readthedocs.io/) for fuzzing

```python
import pytest
from hypervisor.rings.engine import RingEngine, ExecutionRing

class TestRingEngine:
    """Tests for execution ring assignment logic."""

    def test_high_trust_agent_gets_ring_2(self) -> None:
        engine = RingEngine()
        ring = engine.compute_ring(sigma_raw=0.85)
        assert ring == ExecutionRing.RING_2_STANDARD

    def test_untrusted_agent_gets_sandbox(self) -> None:
        engine = RingEngine()
        ring = engine.compute_ring(sigma_raw=0.30)
        assert ring == ExecutionRing.RING_3_SANDBOX

    @pytest.mark.asyncio
    async def test_sudo_elevation_with_ttl(self) -> None:
        engine = RingEngine()
        elevated = await engine.elevate(
            agent_did="did:mesh:agent-1",
            target_ring=ExecutionRing.RING_1_PRIVILEGED,
            ttl_seconds=60,
        )
        assert elevated.ring == ExecutionRing.RING_1_PRIVILEGED
```

### Coverage Requirements

- **Minimum coverage:** 80% for new code
- **Critical paths** (rings, saga, security): aim for 90%+
- Run `python -m pytest --cov=src/hypervisor --cov-report=term-missing` to identify uncovered lines

### Test Organization

```
tests/
├── unit/                         # Fast, isolated unit tests
├── integration/                  # Cross-module integration tests
├── test_agent_manager.py         # Agent lifecycle tests
├── test_breach_detector.py       # Ring breach detection tests
├── test_classifier.py            # Trust score classification tests
├── test_kill_switch.py           # Kill switch & rate limiter tests
├── test_rate_limiter.py          # Per-ring rate limiting tests
├── test_shapley_attribution.py   # Fault attribution tests
└── __init__.py
```

---

## 📋 Issue Guidelines

### Bug Reports

When filing a bug, include:

1. **Agent Hypervisor version** (`pip show agent-hypervisor`)
2. **Python version** (`python --version`)
3. **Operating system**
4. **Steps to reproduce** — minimal code snippet or CLI commands
5. **Expected behavior** vs. **actual behavior**
6. **Full error traceback** if applicable
7. **Which module** is affected (rings, saga, liability, etc.)

### Feature Requests

For feature requests, describe:

1. **Use case** — what governance problem does this solve?
2. **Proposed solution** — how should it work?
3. **Alternatives considered** — what else did you look at?
4. **Which module** should this belong to (or is it a new module)?

### Good First Issues

New to the project? Look for issues labeled:

| Label | Description |
|-------|-------------|
| [`good first issue`](https://github.com/imran-siddique/agent-hypervisor/labels/good%20first%20issue) | Small, well-defined tasks for newcomers |
| [`documentation`](https://github.com/imran-siddique/agent-hypervisor/labels/documentation) | Improve docs, examples, and guides |
| [`help wanted`](https://github.com/imran-siddique/agent-hypervisor/labels/help%20wanted) | Community contributions welcome |

---

## 🏗️ Architecture Overview

### Project Structure

```
agent-hypervisor/
├── src/hypervisor/           # Main package
│   ├── session/              # Shared Session Object lifecycle + VFS
│   ├── rings/                # 4-ring privilege model + elevation + breach detection
│   ├── saga/                 # Saga orchestrator + fan-out + checkpoints + DSL
│   ├── liability/            # Sponsorship, penalty, attribution, quarantine, ledger
│   ├── audit/                # Delta engine, audit log, GC, commitment
│   ├── security/             # Rate limiter, kill switch
│   ├── observability/        # Event bus, causal trace IDs
│   ├── reversibility/        # Execute/Undo API registry
│   ├── verification/         # DID transaction history verification
│   ├── integrations/         # Nexus, Verification, IATP cross-module adapters
│   ├── api/                  # FastAPI REST endpoints (22 endpoints)
│   ├── core.py               # Core utilities and shared logic
│   ├── models.py             # Pydantic models
│   └── providers.py          # Dependency injection providers
├── tests/                    # Test suite (457+ tests)
├── benchmarks/               # Performance benchmarks
├── examples/                 # Working demos and dashboard
├── tutorials/                # Step-by-step tutorials
├── notebooks/                # Jupyter notebooks
└── docs/                     # Documentation
```

### Key Modules

| Module | Purpose | Key Abstractions |
|--------|---------|------------------|
| **session** | Session lifecycle, VFS namespacing, DID-bound identity | `SharedSessionObject`, `SessionConfig`, `VFS` |
| **rings** | 4-ring privilege levels, sudo elevation, breach detection | `RingEngine`, `ExecutionRing`, `BreachDetector` |
| **saga** | Multi-step transactions with compensation & fan-out | `SagaOrchestrator`, `SagaStep`, `CompensationPolicy` |
| **liability** | Sponsorship, penalty, fault attribution, quarantine | `LiabilityEngine`, `VouchChain`, `SlashCascade` |
| **audit** | Hash-chained delta trails, semantic diffs, GC | `DeltaEngine`, `AuditLog`, `Commitment` |
| **security** | Kill switch, per-ring rate limiting | `KillSwitch`, `RateLimiter` |
| **observability** | Structured event bus, causal trace IDs | `EventBus`, `CausalTraceId` |
| **reversibility** | Execute/undo API registry for saga steps | `ReversibilityRegistry` |

### Execution Rings

The privilege model is inspired by hardware CPU rings:

```
Ring 0 (Root)       — Hypervisor config & penalty — requires SRE Witness
Ring 1 (Privileged) — Non-reversible actions — requires eff_score > 0.95 + consensus
Ring 2 (Standard)   — Reversible actions — requires eff_score > 0.60
Ring 3 (Sandbox)    — Read-only / research — default for unknown agents
```

Agents start in Ring 3 and earn promotion based on trust score. Ring breach detection triggers immediate demotion. Sudo elevation provides temporary privilege with TTL.

### Saga Compensation

The saga orchestrator manages multi-step workflows:

- **Timeout enforcement** — steps that hang are automatically cancelled
- **Retry with backoff** — transient failures retry with exponential delay
- **Reverse-order compensation** — on failure, all committed steps are undone
- **Escalation** — if compensation fails, Joint Liability penalty is triggered
- **Parallel execution** — ALL_MUST_SUCCEED / MAJORITY / ANY policies

### Kill Switch & Rate Limiting

The security module provides runtime safety:

- **Kill switch** — graceful termination with saga step handoff to substitute agents
- **Rate limiting** — per-agent, per-ring limits (sandbox: 5 rps, root: 100 rps)
- **Circuit breakers** — automatic protection against cascading failures

---

## 🎯 Design Philosophy

**"Isolation by Default"** — Every agent session is isolated, every action is governed, every violation is attributable.

### We ✅ Want

- Hardware-inspired privilege isolation (execution rings)
- Graduated trust, not binary allow/deny
- Automatic compensation for failed multi-step workflows
- Tamper-evident audit trails (hash-chained deltas)
- Runtime kill switch with graceful handoff
- Composability with Agent OS, Agent Mesh, and Agent SRE

### We ❌ Avoid

- Implicit trust between agents
- Unaudited agent actions
- Static, binary access control
- Data loss on agent termination
- Monolithic design — modules should work standalone

---

## 💬 Getting Help

- **Questions?** Open a [Discussion](https://github.com/imran-siddique/agent-hypervisor/discussions)
- **Found a bug?** Open an [Issue](https://github.com/imran-siddique/agent-hypervisor/issues)
- **Security issue?** See [SECURITY.md](SECURITY.md)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
