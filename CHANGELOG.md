# Changelog

All notable changes to Agent Hypervisor will be documented in this file.

## [1.0.0] — 2026-02-20

### Added
- **Core Hypervisor** orchestrator with session lifecycle management
- **Shared Session Object (SSO)** with VFS, snapshots, and consistency modes
- **4-Ring Execution Model** (Ring 0 Root → Ring 3 Sandbox) based on σ_eff trust scores
- **Joint Liability Engine** with vouching, bonding, and proportional slashing
- **Saga Orchestrator** with step timeouts, retries, and reverse-order compensation
- **Merkle-Chained Audit** with delta capture, commitment engine, and ephemeral GC
- **Reversibility Registry** for execute/undo API mapping with 4 reversibility levels
- **Transaction History Verifier** for DID-based trust verification
- **Integration Adapters** (Protocol-based, zero hard dependencies):
  - Nexus adapter — trust score resolution and caching
  - CMVK adapter — behavioral drift detection with severity thresholds
  - IATP adapter — capability manifest parsing and trust hints
- **184 tests** (unit, integration, and scenario tests)
- **Performance benchmarks** (268μs full pipeline)
- **Interactive demo** (`examples/demo.py`) showcasing all 5 subsystems
- Extracted from [Agent OS](https://github.com/imran-siddique/agent-os) as standalone package
