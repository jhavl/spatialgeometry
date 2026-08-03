# Changelog

Notable changes to this project are documented in this file. Starts from
v1.2.0 — earlier releases aren't documented retroactively.

## [1.3.0] - 2026-08-03

### New

- **`Path` shape**: a polyline through a sequence of waypoints — straight
  segments joining consecutive points, for drawing paths and trajectories
  in the scene. `radius`/`linewidth` mirror `Arrow`'s existing mutual
  exclusivity (a real tube when `radius > 0`, a line honoring `linewidth`
  otherwise). Deliberately a plain `Shape`, not `CollisionShape` — Coal has
  no native "tube along a polyline" primitive.
- **`Axes` gains `arrows=False`** (render each axis as a colored `Arrow`
  instead of a plain line) plus `radius`/`linewidth`, passed straight
  through to each constituent `Arrow` when `arrows=True`.
- **`Arrow` gains `linewidth`**: when `radius == 0` (the existing
  "shaft is a line" fallback), `linewidth` now actually controls that
  line's width in pixels — previously uncontrolled, whatever the browser's
  default happened to render.
- **`__version__`** is now exposed, matching the other packages in the
  toolbox family (`spatialmath-python`, `robotics-toolbox-python`, `bdsim`,
  `machinevision-toolbox-python`).

### Changed

- **`Shape` and `CollisionShape` are now abstract base classes.** Neither
  was meant to be instantiated directly — `Shape.to_dict` and
  `CollisionShape._init_coal` are `@abstractmethod`; every concrete shape
  already overrode both.
- **`T` (pose) moved from `Shape` up to `SceneNode`.** It was a thin
  wrapper with nothing `Shape`-specific about it; moving it up means
  `SceneGroup` (and any future `SceneNode` subclass) gets the same
  convenient, `SE3`-accepting pose property that only `Shape` had before —
  so a whole group of shapes can now be moved as one unit via `group.T`.
- **`repr()`/`str()` rewritten.** `repr()` previously showed only the
  shape's `stype` string plus translation — not unambiguous (`Box`, a
  deprecated alias of `Cuboid`, was indistinguishable from `Cuboid`) and
  broke visually the moment more than one shape appeared together (e.g. in
  a list, or a `SceneGroup`). Now single-line and constructor-style, e.g.
  `Cylinder(radius=1.0, length=2.0, pose='t = 0, 0, 0; rpy/zyx = 0°, 0°, 0°')`.
- **Type hints modernized** across the scene-graph/shape classes —
  `typing.Union`/`List`/`Dict`/`Tuple`/`Optional` replaced with
  `X | Y`/`list`/`dict`/`tuple`, and filled in on several public
  constructors/properties that had none at all.
- Docs: the Sphinx build's `docs` extra now also installs `collision`, so
  the intro chapter's `closest_point()`/`iscollided()` example actually
  runs instead of silently baking a traceback into the rendered page.
- Docs: intro chapter rewritten with mesh/collision worked examples, mouse
  navigation controls, and scene-graph parenting; `sphinx-copybutton`,
  `sphinx-codeautolink`, `sphinxcontrib-mermaid`, and a `make livehtml`
  target wired up for faster doc iteration.

### Fixed

- **`Shape.color`'s setter left an all-integer color input as
  `numpy.int64`** (e.g. `color=[1, 0, 0, 1]`, a very natural way to write
  "opaque red") rather than a plain `float` — `self.color[3]` (opacity)
  being `int64` broke real (non-mocked) `json.dumps()` sends. Never caught
  by the existing test suite since those tests are `FakeBrowser`-mocked.
- **`SceneNode._T`'s setter referenced a nonexistent `self.parent.wT`**
  (should have been `self._scene_parent._wT`) — setting a node's pose
  while it already had a `scene_parent` raised `AttributeError` instead of
  composing the world transform. No test coverage existed for this path
  before.
- `SceneNode.scene_parent`'s return type was annotated `Type["SceneNode"]`
  (the class itself) rather than an instance; corrected to
  `SceneNode | None`.

## [1.2.0] - 2026-07-28

First release since v1.1.0 (2023) — mostly packaging and build-system work, plus a collision-backend swap.

### New

- **Coal replaces PyBullet as the collision backend.** PyBullet wouldn't build with clang on macOS, so all collision checking now runs on [Coal](https://github.com/coal-library/coal) instead.
- **NumPy 2 support.**
- **Pyodide (WebAssembly) wheels**, so spatialgeometry can run in-browser (e.g. JupyterLite), alongside the usual platform wheels. Since there's no C++ compiler to target under Pyodide, that wheel ships a pure-Python fallback for the scene-graph transform math instead of the compiled extension — same API, just not compiled.
- **Prebuilt wheels for more platforms**: native Linux arm64 and macOS arm64 in addition to Linux/Windows x86_64 — no compiler needed on install for any of them.

### Changed

- **Build system rewritten**: the C++ extension now builds with **nanobind + scikit-build-core** (previously a hand-rolled hatchling build hook). Faster builds, and the vendored Eigen copy was trimmed from Eigen's full 337-file tree down to the ~175 files this project actually uses.
- **Project layout moved to `src/`**, and the build backend moved from setuptools to hatchling and then to scikit-build-core over the course of this work.
- **Docs rewritten**: the README, intro chapter, and API reference no longer borrow content from the Robotics Toolbox docs — they now describe spatialgeometry on its own terms, with a Swift-based display example.
- **Dependency cleanup**: `roboticstoolbox` is no longer pulled into the `dev` extra, removing a version-pin collision some installs were hitting.

### Fixed

- macOS build failures caused by PyBullet + clang (resolved by the Coal switch above).
- NumPy 2 compatibility issues.
- Windows wheel builds failing over an overly long temp-file path.
- Several wheel-tagging/build-hook bugs that could produce a stale or wrong-architecture compiled extension, or silently fall back to a non-compiled wheel.
- The PyPI publish pipeline itself: a workflow bug was letting the Pyodide/wasm32 wheel — which PyPI doesn't accept — get bundled into the same upload batch as the real wheels, aborting the upload partway through. Publishing now uses a single, unambiguous trigger (creating a GitHub Release) that always completes the full publish.

### Known limitations

- Collision checking (Coal) isn't available via `pip` on Windows — one of Coal's own dependencies doesn't publish a Windows wheel. Everything else works there; `pip install spatialgeometry[collision]` will just have no effect on that platform for now.
  **Workaround:** conda-forge publishes `coal-python` for `win-64` against its own assimp build, so `conda install -c conda-forge coal-python` followed by `pip install spatialgeometry trimesh` gets collision checking working on Windows today.
