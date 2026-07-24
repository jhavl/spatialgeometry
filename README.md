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

> Collision functionality depends on [Coal](https://github.com/coal-library/coal)
> and is not expected to work in JupyterLite.

For collision support:

```shell
pip install spatialgeometry[collision]
```

This installs [Coal](https://github.com/coal-library/coal) and
[trimesh](https://trimesh.org), both of which publish prebuilt wheels for
macOS (including Apple Silicon/arm64), Linux, and Windows — no conda or
manual build steps required.

### conda / conda-forge

For development, the provided `environment.yml` installs the package in
editable mode with all dev and collision extras:

```shell
conda env create -f environment.yml
conda activate spatialgeometry-dev
```
