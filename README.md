# Spatial Geometry

[![A Python Robotics Package](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/master/.github/svg/py_collection.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)

[![PyPI version](https://badge.fury.io/py/spatialgeometry.svg)](https://badge.fury.io/py/spatialgeometry)
[![Anaconda version](https://anaconda.org/conda-forge/spatialgeometry/badges/version.svg)](https://anaconda.org/conda-forge/spatialgeometry)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spatialgeometry.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Build Status](https://github.com/jhavl/spatialgeometry/workflows/build/badge.svg?branch=main)](https://github.com/jhavl/spatialgeometry/actions?query=workflow%3Abuild)
[![codecov](https://codecov.io/gh/jhavl/spatialgeometry/branch/main/graph/badge.svg?token=YPmchbQi2v)](https://codecov.io/gh/jhavl/spatialgeometry)

<table style="border:0px">
<tr style="border:0px">
<td style="border:0px">
<img src="https://github.com/petercorke/robotics-toolbox-python/raw/master/docs/figs/RobToolBox_RoundLogoB.png" width="200"></td>
<td style="border:0px">
A Python Shape and Geometry Package
<ul>
<li><a href="https://github.com/jhavl/spatialgeometry">GitHub repository </a></li>
<li><a href="https://jhavl.github.io/spatialgeometry">Documentation</a></li>
</ul>
</td>
</tr>
</table>

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

> Collision functionality depends on PyBullet and is not expected to work in
> JupyterLite.

For collision support (requires [PyBullet](https://pybullet.org)):

```shell
pip install spatialgeometry[collision]
```

> **Note for macOS (Apple Silicon / arm64):** PyBullet cannot be built from
> source on macOS with recent Xcode/clang toolchains.  Install it via
> conda-forge **before** installing the collision extra (see below).

### conda / conda-forge

The recommended approach for development, and the only reliable way to get
PyBullet on macOS/arm64, is to use the provided `environment.yml`:

```shell
conda env create -f environment.yml
conda activate spatialgeometry-dev
```

This creates an environment named `spatialgeometry-dev` with:
- numpy ≥ 2.0 (from conda-forge)
- pybullet pre-built binary (from conda-forge – avoids the macOS/clang build issue)
- the package itself installed in editable mode with all dev and collision extras

Alternatively, install pybullet manually from conda-forge into an existing environment before using pip:

```shell
conda install -c conda-forge pybullet
pip install spatialgeometry[collision]
```
