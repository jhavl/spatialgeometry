"""Custom hatchling build hook to compile the C extension (spatialgeometry.scene)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from setuptools import Distribution
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        root = Path(self.root)
        src_pkg = root / "src" / "spatialgeometry"

        ext = Extension(
            name="spatialgeometry.scene",
            sources=[str(src_pkg / "core" / "scene.cpp")],
            include_dirs=[
                str(src_pkg / "core"),
                numpy.get_include(),
            ],
        )

        dist = Distribution({"ext_modules": [ext]})
        cmd = build_ext(dist)
        cmd.ensure_finalized()
        cmd.build_lib = str(root / "src")
        cmd.build_temp = str(root / "build" / "temp")
        cmd.inplace = False
        cmd.run()

        # Register the compiled extension so hatchling includes it in the wheel.
        for so in src_pkg.glob("scene*.so"):
            build_data["artifacts"].append(str(so.relative_to(root)))
        for pyd in src_pkg.glob("scene*.pyd"):
            build_data["artifacts"].append(str(pyd.relative_to(root)))
