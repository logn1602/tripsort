# Decision log

Items from the "Open decisions" list in [SPEC.md](SPEC.md) once they are settled, plus changes
to the plan. Newest last.

| Date | Decision | Notes |
| --- | --- | --- |
| 2026-11-17 | Package name: `tripsort` | Free on PyPI as of 2026-11-17. |
| 2026-11-17 | Project license: MIT | Settles "MIT or Apache-2.0" before Phase 1. |
| 2026-11-17 | Phase 1 publishes the 0.0.1 stub to PyPI as well as TestPyPI | The spec planned TestPyPI only for Phase 1 and PyPI from Phase 6. Every `v*` tag on `main` now goes to TestPyPI, is install-checked there, then goes to PyPI. |
| 2026-11-17 | Phase 1 runtime dependencies: `typer` only | The other core dependencies are added in the phase that first uses them. |
