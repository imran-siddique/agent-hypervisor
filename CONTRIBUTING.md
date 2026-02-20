# Contributing to Agent Hypervisor

Thank you for your interest in contributing! We welcome contributions of all kinds.

## Getting Started

```bash
git clone https://github.com/imran-siddique/agent-hypervisor.git
cd agent-hypervisor
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Run linting: `ruff check src/ tests/`
6. Commit with [conventional commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, etc.
7. Open a Pull Request

## Architecture

The hypervisor is organized into 7 subsystems + integration adapters:

| Module | Purpose |
|--------|---------|
| `session/` | Shared Session Object (SSO) lifecycle |
| `rings/` | 4-ring execution privilege model |
| `liability/` | Vouching, bonding, collateral slashing |
| `reversibility/` | Execute/Undo API mapping |
| `saga/` | Semantic saga with compensation |
| `audit/` | Merkle-chained delta engine |
| `verification/` | DID transaction history |
| `integrations/` | Nexus, CMVK, IATP adapters |

## Code Style

- Python 3.11+
- Type hints on all public APIs
- Ruff for linting (`ruff check`)
- MyPy for type checking (`mypy src/`)
- Max line length: 100

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Benchmarks
python benchmarks/bench_hypervisor.py
```

## Related Projects

- [Agent OS](https://github.com/imran-siddique/agent-os) — Governance kernel
- [Agent Mesh](https://github.com/imran-siddique/agent-mesh) — Trust network
- [Agent SRE](https://github.com/imran-siddique/agent-sre) — Reliability platform
