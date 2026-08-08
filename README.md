# Spatial Geometry

[![A Python Robotics Package](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/main/.github/svg/py_collection.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)

[![PyPI version](https://badge.fury.io/py/spatialgeometry.svg)](https://badge.fury.io/py/spatialgeometry)
[![Anaconda version](https://anaconda.org/conda-forge/spatialgeometry/badges/version.svg)](https://anaconda.org/conda-forge/spatialgeometry)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spatialgeometry.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Build Status](https://github.com/jhavl/spatialgeometry/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/jhavl/spatialgeometry/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/jhavl/spatialgeometry/branch/main/graph/badge.svg?token=YPmchbQi2v)](https://codecov.io/gh/jhavl/spatialgeometry)

[GitHub repository](https://github.com/jhavl/spatialgeometry) &nbsp;|&nbsp; [Documentation](https://jhavl.github.io/spatialgeometry)

Spatial Geometry provides simple 3D shape primitives -- cuboids, cylinders,
spheres, and triangle meshes -- for representing robot links, obstacles, and
other geometry in a scene. Every shape carries a pose (position and
orientation) and an optional colour, and can be tested for distance and
collision against any other shape using [Coal](https://github.com/coal-library/coal).

It's used by the [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)
to describe robot link geometry, and by [Swift](https://github.com/jhavl/swift)
to render scenes in the browser.

## Quick start

```python
import spatialgeometry as gm
from spatialmath import SE3

cube = gm.Cuboid([1, 1, 1], pose=SE3(0, 0, 0))
sphere = gm.Sphere(0.5, pose=SE3(2, 0, 0), color="red")

d, p1, p2 = cube.closest_point(sphere, inf_dist=10)
cube.iscollided(sphere)
```

See the [documentation](https://jhavl.github.io/spatialgeometry) for the
full API reference and more examples, including how to display shapes with
Swift.

## Installation

### pip (standard)

```shell
pip install spatialgeometry
```

### JupyterLite / Pyodide

For browser runtimes (JupyterLite/Pyodide), use a pure-Python wheel build:

```shell
SPATIALGEOMETRY_BUILD_EXTENSION=0 python -m build --wheel
```

This disables native C-extension compilation and produces a wheel that uses the
Python scene backend (`spatialgeometry.scene`).

> Collision functionality depends on [Coal](https://github.com/coal-library/coal)
> and is not expected to work in JupyterLite.

For collision support:

```shell
pip install spatialgeometry[collision]
```

This installs [Coal](https://github.com/coal-library/coal) and
[trimesh](https://trimesh.org). Coal publishes prebuilt wheels for macOS
(including Apple Silicon/arm64) and Linux — no conda or manual build steps
required there.

> **Windows note:** Coal doesn't currently install via pip on Windows (one
> of its own dependencies, `cmeel-assimp`, has no Windows wheel in the
> version range Coal needs), so the `collision` extra skips it there and
> collision checking is unavailable via `pip` on Windows. Everything else
> in the package works normally.
>
> Workaround: conda-forge publishes `coal-python` built for `win-64`
> against its own assimp, sidestepping the missing wheel entirely:
>
> ```shell
> conda install -c conda-forge coal-python
> pip install spatialgeometry trimesh
> ```

If you don't need collision checking but do want to work with mesh files --
e.g. `Mesh.corners()`/`bounds()`/`extents()` for a mesh's bounding box, which
reads the file via [trimesh](https://trimesh.org) but never touches Coal at
all -- install just that piece:

```shell
pip install spatialgeometry[mesh]
```

Unlike `collision`, this works on Windows too: trimesh has no Windows-wheel
problem of its own, only Coal does.

### conda / conda-forge

For development, the provided `environment.yml` installs the package in
editable mode with all dev and collision extras:

```shell
conda env create -f environment.yml
conda activate spatialgeometry-dev
```

## License

MIT — see [LICENSE](LICENSE).
