# spatialgeometry — Agent Instructions

Part of the RVC ecosystem. **Read [rvc-ecosystem/AGENTS.md](https://github.com/petercorke/rvc-ecosystem/blob/main/AGENTS.md) first** — it defines shared conventions: repo ownership, math invariants, dependency boundaries, git/PR workflow, code standards, tech-debt tracking. This file only adds what's specific to this repo.

| | |
|---|---|
| PyPI package | `spatialgeometry` |
| Nickname | SG |
| Owner | Jesse Haviland (`jhavl`) — **not Peter Corke** |
| Default branch | `main` |
| Contribution model | **PR only** |

## Notes specific to this repo

- **This is not Peter's repo.** Every change goes through a PR, even trivial docs fixes;
  merging is the owner's call, and `main` is branch-protected. Jesse confirmed (2026-09-14)
  that branches may be pushed straight to `origin` and the PR opened from there, so no fork
  is needed. Pushing to a personal `fork` remote instead is equally fine. Never push to
  `main` directly.
- Compiles `nanobind` C++ extensions via `scikit-build-core`.
- Depends on `spatialmath` only. `robotics-toolbox-python` and `swift` both depend on this
  repo — a breaking API change here has two downstream repos to check.
- Default branch is `main` — this repo has already migrated off `master`; don't assume
  otherwise from old references.
