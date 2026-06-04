"""Custom hatchling build hook to compile the C extension (spatialgeometry.scene)."""

from __future__ import annotations

import os
import platform
import tempfile
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

        # On Windows, MSVC has a 260-char MAX_PATH limit.  The object file path
        # is build_temp/<absolute-source-path>/scene.obj, which easily exceeds
        # the limit when pip unpacks into a deep temp directory.  Use a short
        # system temp dir to keep the path well under the limit.
        if platform.system() == "Windows":
            build_temp = tempfile.mkdtemp(prefix="sg_", dir="C:\\")
        else:
            build_temp = str(root / "build" / "temp")

        try:
            dist = Distribution({"ext_modules": [ext]})
            cmd = build_ext(dist)
            cmd.ensure_finalized()
            cmd.build_lib = str(root / "src")
            cmd.build_temp = build_temp
            cmd.inplace = False
            cmd.force = True  # Always recompile; don't reuse stale pre-committed .so files
            cmd.run()
        except Exception as exc:
            strict_ext = os.getenv("SPATIALGEOMETRY_STRICT_EXTENSION", "0").lower()
            if strict_ext in {"1", "true", "yes"}:
                raise

            warn(
                "Native extension build failed; continuing with pure-Python wheel. "
                f"Set SPATIALGEOMETRY_STRICT_EXTENSION=1 to fail instead. ({exc})"
            )
            # Explicitly include the pure-Python fallback so hatchling doesn't
            # accidentally omit it (e.g. if artifact handling strips it).
            scene_py = src_pkg / "scene.py"
            build_data["force_include"][str(scene_py)] = "spatialgeometry/scene.py"
            return

        # Include only the exact .so/.pyd that setuptools just compiled for this
        # Python version/platform.  Using a glob would accidentally pick up other
        # pre-committed binaries (wrong arch or wrong Python version).
        ext_rel = cmd.get_ext_filename(ext.name)   # e.g. "spatialgeometry/scene.cpython-310-darwin.so"
        so_path = Path(cmd.build_lib) / ext_rel
        if not so_path.exists():
            raise RuntimeError(f"Expected compiled extension not found: {so_path}")
        build_data["force_include"][str(so_path)] = "spatialgeometry/" + so_path.name

        # Tell hatchling to tag this as a platform wheel, not py3-none-any.
        # pure_python=False sets Root-Is-Purelib; infer_tag=True makes hatchling
        # call get_best_matching_tag() to produce the correct cpXY-cpXY-platform tag.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
