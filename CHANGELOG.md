# Changelog

Notable changes to this project are documented in this file. Starts from
v1.2.0 — earlier releases aren't documented retroactively.

## [1.4.0] - 2026-08-09

### New

- **`Ellipsoid` shape**: a native `coal.Ellipsoid` collision primitive,
  parameterised as `radii=[rx, ry, rz]` (three semi-axis lengths), following
  the same pattern as `Sphere`/`Cylinder`/`Cuboid`.
- **`Shape.corners()`/`bounds()`/`extents()`** for a shape's axis-aligned
  bounding box — `corners(world=False)`, `bounds(world=False)`,
  `extents(world=False)`; pass `world=True` for the posed shape's envelope
  in the world frame. Implemented for `Cuboid`/`Box`, `Sphere`, `Cylinder`,
  `Mesh`; `Axes`/`Arrow`/`Polyline` raise `NotImplementedError` for now rather
  than a silently wrong value.
- **`SceneNode.tree()`/`tree_children()`**: render a scene graph (or just
  one node's own subtree) as human-readable indented text, one `repr()` per
  line, with `tree()` marking the calling node's position with a trailing
  `<==`. `SceneGroup` needs no special-casing — the tree walk is generic.
- **`CollisionShapeGroup`**: an ordered, list-like collection of
  `CollisionShape`/`CollisionShapeGroup` instances that itself behaves like
  a single collision-checkable shape — `iscollided()`/`closest_point()` work
  in every combination of shape/group, including arbitrarily nested groups.
  Comes with a new **`&` operator** (`a & b` is `a.iscollided(b)`) that
  works uniformly across shapes and groups.
- **`Shape.opacity` property**: a convenient way to get/set just the alpha
  channel of `color` without needing to know or re-specify the current
  `(r, g, b)`. `set_alpha()` is deprecated in its favor.
- **`Arrow.FromTo(start, end)`**: construct an arrow spanning two 3-vectors
  directly, with `length` and `pose` computed automatically instead of by
  hand.
- Tutorial introduction: substantially expanded, with worked examples
  covering shape creation, bounding boxes, collision checking, scene
  graphs, and groups.

### Mesh

A cluster of related `Mesh` improvements landed together this release:

- **New `y_up` param**, for mesh files authored with +Y as "up" (a common
  convention in general 3D/graphics tooling) rather than this ecosystem's
  +Z-up convention — applies a fixed correction to the mesh's own vertex
  data at load time, so it survives re-posing/animation. `scale` also now
  accepts a single scalar (uniform on all 3 axes), not just a 3-element
  list.
- **New `mesh` extra** (`pip install spatialgeometry[mesh]`): just
  `trimesh`, split out of `collision` — unlike Coal, trimesh has no
  Windows-wheel problem, so mesh-only functionality that never touches Coal
  (`corners()`/`bounds()`/`extents()`, which work independent of
  `collision=True/False`) is now reachable via `pip` on Windows.
  `collision` still pulls in `mesh` transitively, nothing changes for
  existing `collision` users.
- **Now tracks whether an explicit color was ever given**
  (`use_vertex_colors` in `to_dict()`), so a renderer has a signal to
  prefer a mesh file's own baked-in per-vertex/per-face colors over a flat
  default grey when no color was ever actually requested.
- **`filename`/`y_up` are now read-only.** There's no use case for
  repointing an existing `Mesh` at a different file or a different up-axis
  convention — construct a new one instead. (Both are also only ever read
  once, inside `_init_coal()`, which is cached after first use — a setter
  would have silently done nothing after a shape's first
  `closest_point()`/`iscollided()` call.)
- **`Mesh(filename=...)` now raises `FileNotFoundError` at construction**
  if the file doesn't exist, instead of deferring the failure to the first
  `closest_point()`/`iscollided()` call.
- **Fixed: `corners()`/`bounds()`/`extents()` now correctly apply the
  `y_up` correction.** Previously the `+Y → +Z` correction was only applied
  to the collision geometry, not the bounding box, so a `y_up=True` mesh's
  bounding box didn't match its actual (corrected) orientation.

### Changed

- **`repr()` now shows `color`, and `opacity` when not fully opaque** —
  previously only subclass-specific params (`radius`, `scale`, ...) and
  `pose` were shown.
- **`Path` shape renamed to `Polyline`**, to avoid clashing with the
  standard library's `pathlib.Path` — a source of confusing type errors in
  code that imports both. `Path` remains available as a deprecated alias
  (`class Path(Polyline)`), emitting a `FutureWarning` on construction.
- **`SceneNode.__init__`'s `T=` kwarg renamed to `pose=`**, now accepting
  `SE3` as well as a raw `ndarray` — matching the convenience `Shape` (and
  every concrete shape) already had. `SceneNode` is the base class for the
  whole scene graph, so this had been implemented one level too low;
  `SceneGroup` previously had no way to accept an `SE3` pose at
  construction, only a raw `ndarray` via `T=`.
- **`update()` is now the public method for refreshing the scene graph's
  world transforms**, replacing `_propogate_scene_tree()` (a straight
  misspelling of "propagate", and not actually private in practice —
  called directly by `robotics-toolbox-python` and `swift`, and
  demonstrated in this project's own tutorial). `_propogate_scene_tree()`
  still works, emitting a `FutureWarning`, for one release cycle.
- **Color-parsing error messages now name matplotlib explicitly** and link
  to its full named-colors reference, rather than a bare "invalid color
  name" with no pointer to where valid names are defined.
- Internal: `src/spatialgeometry/core` renamed to `cpp-extension`, for
  naming consistency — nothing importable changed.
- Docs: substantial API reference cleanup — a new page on mesh file format
  support (which formats work for display vs. collision, and why);
  `autoclass_content='both'` so each class's own constructor parameters
  (previously invisible on most pages, since Sphinx's default only shows
  the class docstring) now actually appear; several stale/incorrect
  passages fixed in the intro tutorial.

### Fixed

- **`Cylinder`'s collision geometry was half its documented length.**
  `_init_coal()` passed `length / 2.0` to Coal on the mistaken assumption
  its constructor took a half-length — it takes the full length, so
  `Cylinder(radius=1, length=4)`'s actual collision geometry was only 2
  units tall. No existing test exercised a cylinder along its own axis, only
  radially, so this went undetected.
- **`scene_parent` assignments that would create a cycle are now rejected**
  (`a.scene_parent = b; b.scene_parent = a`, a self-loop, or a longer
  cycle). Previously nothing validated this, and the scene-graph
  root-finding walk (used by `update()` every call) has no cycle detection
  of its own — a cycle made it spin forever rather than raise.
- **`SceneGroup`'s list mutators (`append`/`extend`/`insert`/`remove`/
  `pop`/`clear`/item assignment) didn't wire `scene_parent`** — elements
  added this way never actually became children in the scene graph, only
  in the list. The constructor also now accepts an initial list
  (`SceneGroup([a, b])` previously raised `TypeError`).
- `repr()` could leak `numpy.float64` values (e.g. `np.float64(1.0)`
  instead of `1.0`) into `color`/`opacity` — harmless functionally, ugly in
  a repr.

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
