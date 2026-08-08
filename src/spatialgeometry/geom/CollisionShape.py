#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from typing import Any

import numpy as np
from spatialmath.base.argcheck import getvector
from spatialgeometry.geom import Shape
from spatialgeometry.geom.Shape import ArrayLike, aabb_corners, update
from warnings import warn

# Module-level coal reference — populated on first use, never in Pyodide.
_coal = None


def _require_coal() -> None:
    """Import coal on first use; raise clearly if unavailable."""
    global _coal
    if _coal is not None:
        return
    if sys.platform == "emscripten":
        raise RuntimeError(
            "Collision detection is not available in the browser (Pyodide) "
            "environment. Use a native Python installation for collision checking."
        )
    try:
        import coal as _c
        _coal = _c
    except ImportError:
        raise ImportError(
            "The 'coal' package is required for collision functionality. "
            "Install with:  pip install coal"
        )


class CollisionShape(Shape):
    def __init__(self, collision: bool = True, **kwargs) -> None:
        self.co = None      # coal.CollisionObject, created on first use
        self._cinit = False
        super().__init__(**kwargs)
        self._collision = collision

    def _update_coal(self) -> None:
        """Push the current world transform into the Coal collision object."""
        if self.co is not None:
            self.co.setTranslation(self._wT[:3, 3])
            self.co.setRotation(self._wT[:3, :3])

    @abstractmethod
    def _init_coal(self) -> None:
        """Build this shape's Coal collision geometry. Implemented by each concrete subclass."""

    def _ensure_coal(self) -> None:
        """Guarantee Coal is loaded and this object's Coal twin is current."""
        _require_coal()
        if not self._cinit:
            self._init_coal()
        self._update_coal()

    def closest_point(
        self, shape: CollisionShape, inf_dist: float = 1.0
    ) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        """
        Return the minimum euclidean distance between self and shape.

        :param shape: The shape to compare distance to
        :param inf_dist: Only return a result when distance < inf_dist
        :returns: (d, p1, p2) — distance and closest points in world frame,
            or (None, None, None) when the shapes are farther than inf_dist.
            d is negative when the shapes are penetrating.
        """
        self._ensure_coal()
        shape._ensure_coal()

        req = _coal.DistanceRequest()
        req.enable_signed_distance = True
        res = _coal.DistanceResult()
        _coal.distance(self.co, shape.co, req, res)

        d = res.min_distance
        if d > inf_dist:
            return None, None, None
        return d, np.array(res.getNearestPoint1()), np.array(res.getNearestPoint2())

    def iscollided(self, shape: CollisionShape) -> bool:
        """
        Return True if self and shape have collided (distance ≤ 0).

        :param shape: The shape to check against
        """
        d, _, _ = self.closest_point(shape)
        return d is not None and d <= 0

    def collided(self, shape: CollisionShape) -> bool:
        """Deprecated — use iscollided instead."""
        warn("collided is deprecated, use iscollided instead", FutureWarning)
        return self.iscollided(shape)


# =====================================================================
# LOUD WARNING -- read this before touching y_up.
#
# This rotation MUST stay bit-for-bit equivalent to the y_up correction
# applied in Swift's shapes.js (search "y_up" there -- applied via
# geometry.rotateX() on the loaded THREE.Geometry, once, at load time).
#
# These two implementations live in different languages in different
# repos and NOTHING enforces they agree. If they ever diverge, there is
# no test or type checker that will catch it -- collision geometry will
# just silently stop matching what's actually rendered. If you change
# one side, you MUST change the other, in the same PR pair.
#
# What it does: a mesh authored with +Y as "up" is reinterpreted as if
# it were authored +Z "up" (this ecosystem's convention). Equivalent to
# Rx(+90 degrees) applied to the mesh's own local vertex data:
# +Y -> +Z, +Z -> -Y, +X unchanged. Baked into the static vertex data
# itself (here, once, at load time) rather than into the shape's live
# pose -- so it survives re-posing/animation, same reasoning as Swift's
# existing Cylinder axis correction (three.js's CylinderGeometry
# defaults to axis-along-Y; shapes.js corrects it the same way).
# =====================================================================
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


class Mesh(CollisionShape):
    """
    A mesh object described by an STL, OBJ, or DAE file.

    :param filename: Absolute path to the mesh file.
    :param scale: Scale factor(s) along XYZ axes (default [1, 1, 1]). A
        single number applies the same scale to all three axes.
    :param y_up: Set True if the mesh file was authored with +Y as the
        "up" axis -- a common convention in general 3D/graphics tooling --
        rather than this ecosystem's +Z-up convention. See the
        ``_Y_UP_TO_Z_UP`` comment in this module for the full story; the
        short version is that Swift applies the matching correction on
        its own side, and the two must be kept in sync.
    :param collision: Whether this shape participates in collision checking.
    """

    _repr_params = ("filename", "scale", "y_up")

    def __init__(
        self,
        filename: str | None = None,
        scale: ArrayLike | float = [1, 1, 1],
        y_up: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(stype="mesh", **kwargs)
        self.filename = filename
        self.scale = scale
        self.y_up = y_up

    def _init_coal(self) -> None:
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "The 'trimesh' package is required for mesh collision objects. "
                "Install with:  pip install trimesh"
            )

        mesh = trimesh.load(self.filename, force="mesh")
        vertices = mesh.vertices
        if self.y_up:
            # See the LOUD WARNING on _Y_UP_TO_Z_UP above -- Swift's
            # shapes.js must apply the identical correction.
            vertices = vertices @ _Y_UP_TO_Z_UP
        vertices = (vertices * self.scale).astype(np.float64, order="C")
        triangles = mesh.faces.astype(np.int64, order="C")

        bvh = _coal.BVHModelOBBRSS()
        bvh.beginModel(len(triangles), len(vertices))
        bvh.addVertices(vertices)
        bvh.addTriangles(triangles)
        bvh.endModel()

        self.co = _coal.CollisionObject(bvh)
        self._cinit = True

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    @update
    def scale(self, value: ArrayLike | float | None) -> None:
        if value is None:
            value = [1, 1, 1]
        elif np.isscalar(value):
            value = [value, value, value]
        self._scale = np.array(getvector(value, 3))

    @property
    def filename(self) -> str | None:
        return self._filename

    @filename.setter
    @update
    def filename(self, value: str | None) -> None:
        self._filename = value

    @property
    def y_up(self) -> bool:
        """
        True if this mesh file was authored with +Y as "up" and needs
        the +Y -> +Z correction applied. See the ``_Y_UP_TO_Z_UP`` LOUD
        WARNING comment above ``Mesh`` -- Swift applies the matching
        correction on its own side, and the two must stay in sync.

        This is a read/write property.

        :rtype: bool
        """
        return self._y_up

    @y_up.setter
    @update
    def y_up(self, value: bool) -> None:
        self._y_up = bool(value)

    def to_dict(self) -> dict[str, Any]:
        shape = super().to_dict()
        shape["filename"] = self.filename
        shape["scale"] = self.scale.tolist()
        shape["y_up"] = self.y_up
        return shape

    def _local_corners(self) -> np.ndarray:
        # Independent of self.collision/_init_coal() -- a mesh's bounding
        # box is a plain geometric fact, not a collision-only concern, so
        # this loads via trimesh itself rather than reusing _init_coal()
        # (which raises ValueError when collision=False).
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "The 'trimesh' package is required to compute a Mesh's "
                "bounding box. Install with:  pip install trimesh"
            )
        mesh = trimesh.load(self.filename, force="mesh")
        mn, mx = mesh.bounds
        return aabb_corners(mn * self.scale, mx * self.scale)


class Cylinder(CollisionShape):
    """
    A cylinder whose centre is at the local origin, axis along Z.

    :param radius: Radius in metres.
    :param length: Total length in metres.
    :param collision: Whether this shape participates in collision checking.
    """

    _repr_params = ("radius", "length")

    def __init__(self, radius: float, length: float, **kwargs) -> None:
        super().__init__(stype="cylinder", **kwargs)
        self.radius = radius
        self.length = length

    def _init_coal(self) -> None:
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        # Coal Cylinder(radius, halfLength)
        geom = _coal.Cylinder(self.radius, self.length / 2.0)
        self.co = _coal.CollisionObject(geom)
        self._cinit = True

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    @update
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    @update
    def length(self, value: float) -> None:
        self._length = float(value)

    def to_dict(self) -> dict[str, Any]:
        shape = super().to_dict()
        shape["radius"] = self.radius
        shape["length"] = self.length
        return shape

    def _local_corners(self) -> np.ndarray:
        r, h = self.radius, self.length / 2.0
        return aabb_corners([-r, -r, -h], [r, r, h])


class Sphere(CollisionShape):
    """
    A sphere whose centre is at the local origin.

    :param radius: Radius in metres.
    :param collision: Whether this shape participates in collision checking.
    """

    _repr_params = ("radius",)

    def __init__(self, radius: float, **kwargs) -> None:
        super().__init__(stype="sphere", **kwargs)
        self.radius = radius

    def _init_coal(self) -> None:
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        self.co = _coal.CollisionObject(_coal.Sphere(self.radius))
        self._cinit = True

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    @update
    def radius(self, value: float) -> None:
        self._radius = float(value)

    def to_dict(self) -> dict[str, Any]:
        shape = super().to_dict()
        shape["radius"] = self.radius
        return shape

    def _local_corners(self) -> np.ndarray:
        r = self.radius
        return aabb_corners([-r, -r, -r], [r, r, r])


class Cuboid(CollisionShape):
    """
    A rectangular prism whose centre is at the local origin.

    :param scale: [length, width, height] in metres.
    :param collision: Whether this shape participates in collision checking.
    """

    _repr_params = ("scale",)

    def __init__(self, scale: ArrayLike, **kwargs) -> None:
        super().__init__(stype="cuboid", **kwargs)
        self.scale = scale

    def _init_coal(self) -> None:
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        s = self.scale
        # Coal Box(x, y, z) takes full dimensions (not half-extents)
        self.co = _coal.CollisionObject(_coal.Box(s[0], s[1], s[2]))
        self._cinit = True

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    @update
    def scale(self, value: ArrayLike) -> None:
        value = getvector(value if value is not None else [1, 1, 1], 3)
        self._scale = np.array(value)

    def to_dict(self) -> dict[str, Any]:
        shape = super().to_dict()
        shape["scale"] = self.scale.tolist()
        return shape

    def _local_corners(self) -> np.ndarray:
        h = self.scale / 2.0
        return aabb_corners(-h, h)


class Box(Cuboid):
    def __init__(self, scale: ArrayLike, **kwargs) -> None:
        warn("Box is deprecated, use Cuboid instead", FutureWarning)
        super().__init__(scale, **kwargs)
