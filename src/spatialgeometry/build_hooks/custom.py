"""Custom hatchling build hook to compile the C extension (spatialgeometry.scene)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from warnings import warn

import numpy
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from setuptools import Distribution
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        build_ext_enabled = os.getenv("SPATIALGEOMETRY_BUILD_EXTENSION", "1").lower()
        if build_ext_enabled in {"0", "false", "no"}:
            return

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

        try:
            dist = Distribution({"ext_modules": [ext]})
            cmd = build_ext(dist)
            cmd.ensure_finalized()
            cmd.build_lib = str(root / "src")
            cmd.build_temp = str(root / "build" / "temp")
            cmd.inplace = False
            cmd.run()
        except Exception as exc:
            strict_ext = os.getenv("SPATIALGEOMETRY_STRICT_EXTENSION", "0").lower()
            if strict_ext in {"1", "true", "yes"}:
                raise

            warn(
                "Native extension build failed; continuing with pure-Python wheel. "
                f"Set SPATIALGEOMETRY_STRICT_EXTENSION=1 to fail instead. ({exc})"
            )
            return

        # Register the compiled extension so hatchling includes it in the wheel.
        for so in src_pkg.glob("scene*.so"):
            build_data["artifacts"].append(str(so.relative_to(root)))
        for pyd in src_pkg.glob("scene*.pyd"):
            build_data["artifacts"].append(str(pyd.relative_to(root)))
