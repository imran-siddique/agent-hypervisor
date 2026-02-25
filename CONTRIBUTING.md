# Contributing to Agent Hypervisor

Thank you for your interest in contributing! We welcome contributions of all
kinds — bug fixes, new features, documentation improvements, and test coverage.

## Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.11+** (3.12 and 3.13 are also supported)
- **git**
- **pip** (bundled with Python)

## Getting Started

### Clone and install

```bash
git clone https://github.com/imran-siddique/agent-hypervisor.git
cd agent-hypervisor
pip install -e ".[dev]"
```

This installs the package in editable mode along with the dev dependencies
(pytest, pytest-asyncio, pytest-cov, hypothesis, ruff, mypy).

### Set up pre-commit hooks

The project uses [pre-commit](https://pre-commit.com/) to enforce code quality
checks automatically:

```bash
pip install pre-commit
pre-commit install
```

Hooks run ruff (lint + format), mypy, and general file checks on every commit.
A pytest gate runs on `git push`.

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ -v --cov=src/hypervisor --cov-report=term-missing
```

## Running the API Server

The optional API layer is powered by FastAPI and Uvicorn. Install the `api`
extra and start the server:

```bash
pip install -e ".[api]"
uvicorn hypervisor.api.server:app
```

By default the server listens on `http://127.0.0.1:8000`. Add `--reload` during
development for auto-restart on file changes.

## Code Style

- **Python 3.11+** with type hints on all public APIs
- **Ruff** for linting and formatting (`ruff check src/ tests/` / `ruff format src/ tests/`)
- **MyPy** for static type checking (`mypy src/`)
- Max line length: **100** characters
- Ruff lint rules: `E, F, I, W, B, C4, UP`

Run the full style check locally before pushing:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
```

## Pull Request Process

1. **Fork** the repository and clone your fork.
2. Create a branch from `main` using the naming convention below.
3. Make your changes in small, focused commits.
4. Ensure all tests pass and linting is clean.
5. Push your branch and open a Pull Request against `main`.
6. Fill in the PR template — describe *what* changed and *why*.
7. A maintainer will review your PR. Please respond to feedback promptly.

### Branch Naming

Use a prefix that describes the type of change:

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `test/` | Adding or updating tests |
| `refactor/` | Code refactoring (no behaviour change) |
| `chore/` | Maintenance tasks (CI, deps, tooling) |

Examples: `feat/session-timeout`, `fix/ring-escalation-bug`, `docs/api-examples`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add session expiry policy
fix: correct ring privilege check for Ring-2
docs: update README with new install steps
```

## Reporting Issues

Found a bug or have a feature request? Please
[open an issue](https://github.com/imran-siddique/agent-hypervisor/issues/new)
and include:

- A clear, descriptive title.
- Steps to reproduce the problem (for bugs).
- Expected vs. actual behaviour.
- Python version and OS.
- Any relevant logs or error output.

For security vulnerabilities, please see [SECURITY.md](SECURITY.md) instead of
opening a public issue.

## Related Projects

- [Agent OS](https://github.com/imran-siddique/agent-os) — Governance kernel
- [Agent Mesh](https://github.com/imran-siddique/agent-mesh) — Trust network
- [Agent SRE](https://github.com/imran-siddique/agent-sre) — Reliability platform
