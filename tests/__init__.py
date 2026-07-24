import importlib.util

import pytest


def _available(*packages):
    return all(importlib.util.find_spec(p) is not None for p in packages)


skip_no_collision_checking = pytest.mark.skipif(
    not _available("coal", "trimesh"),
    reason="coal not installed (pip install '.[collision]')",
)

# roboticstoolbox is a downstream consumer of spatialgeometry, not a
# dependency of it -- deliberately not part of the dev extra (a published
# RTB release vendoring its own spatialgeometry copy at the same import
# path caused real file-clobbering collisions during install). These
# tests only run if a caller has installed it separately, e.g. the
# dedicated CI integration job.
skip_no_rtb = pytest.mark.skipif(
    not _available("roboticstoolbox"),
    reason="roboticstoolbox not installed (optional, for cross-package integration tests)",
)
