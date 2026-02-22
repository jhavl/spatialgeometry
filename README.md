# Spatial Geometry

[![PyPI version](https://badge.fury.io/py/spatialgeometry.svg)](https://badge.fury.io/py/spatialgeometry)
[![Anaconda version](https://anaconda.org/conda-forge/spatialgeometry/badges/version.svg)](https://anaconda.org/conda-forge/spatialgeometry)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spatialgeometry.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)

[![Build Status](https://github.com/jhavl/spatialgeometry/workflows/build/badge.svg?branch=main)](https://github.com/jhavl/spatialgeometry/actions?query=workflow%3Abuild)
[![codecov](https://codecov.io/gh/jhavl/spatialgeometry/branch/main/graph/badge.svg?token=YPmchbQi2v)](https://codecov.io/gh/jhavl/spatialgeometry)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jhavl/spatialgeometry.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jhavl/spatialgeometry/context:python)

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
