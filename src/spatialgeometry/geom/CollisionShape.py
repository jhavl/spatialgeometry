#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections import UserList
from collections.abc import Iterable
from typing import Any

import numpy as np
from spatialmath.base.argcheck import getvector
from spatialgeometry.geom import Shape
from spatialgeometry.geom.Shape import ArrayLike, aabb_corners, mark_changed
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

        :param shape: The shape (or :class:`CollisionShapeGroup`) to
            compare distance to
        :param inf_dist: Only return a result when distance < inf_dist
        :returns: (d, p1, p2) — distance and closest points in world frame,
            or (None, None, None) when the shapes are farther than inf_dist.
            d is negative when the shapes are penetrating.
        """
        if isinstance(shape, CollisionShapeGroup):
            # Delegate to the group's own iteration rather than duplicating
            # it here -- swap p1/p2 back, since shape.closest_point(self,
            # ...) returns them from the group's own point of view (p1 on
            # the group, p2 on self), the opposite of this method's own
            # "p1 on self, p2 on shape" contract.
            d, p_other, p_self = shape.closest_point(self, inf_dist)
            return d, p_self, p_other

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

        :param shape: The shape (or :class:`CollisionShapeGroup`) to check
            against
        """
        if isinstance(shape, CollisionShapeGroup):
            return shape.iscollided(self)
        d, _, _ = self.closest_point(shape)
        return d is not None and d <= 0

    def collided(self, shape: CollisionShape) -> bool:
        """Deprecated — use iscollided instead."""
        warn("collided is deprecated, use iscollided instead", FutureWarning)
        return self.iscollided(shape)

    def __and__(self, other: object) -> bool:
        """
        ``a & b`` is ``a.iscollided(b)`` -- returns a bool, unlike e.g.
        `shapely <https://shapely.readthedocs.io>`_'s ``&``/``intersection()``,
        which returns a geometry. Works for shape-vs-shape, shape-vs-group,
        and (via :class:`CollisionShapeGroup`'s own ``__and__``)
        group-vs-shape and group-vs-group.

        :param other: The shape (or :class:`CollisionShapeGroup`) to check
            against
        """
        if not isinstance(other, CollisionShape):
            return NotImplemented
        return self.iscollided(other)


class CollisionShapeGroup(CollisionShape, UserList):
    """
    An ordered, list-like collection of :class:`CollisionShape` (or nested
    :class:`CollisionShapeGroup`) objects that itself behaves like a single
    collision-checkable shape.

    Unlike :class:`~spatialgeometry.SceneGroup`, which admits any
    :class:`SceneNode`, a :class:`CollisionShapeGroup` only accepts
    :class:`CollisionShape` or :class:`CollisionShapeGroup` instances --
    adding anything else (a plain :class:`Shape` such as :class:`Axes`, for
    instance) raises :exc:`TypeError`.

    ``iscollided()``/``closest_point()`` work in every combination of
    shape-vs-shape, shape-vs-group, group-vs-shape, and group-vs-group,
    including arbitrarily nested groups -- a group checks each of its own
    elements in turn (delegating to that element's own
    ``iscollided()``/``closest_point()``, which handles the other side
    being a group itself if needed) and aggregates: any element collided
    means the group collided; the closest element's own ``(d, p1, p2)`` is
    the group's.

    A :class:`CollisionShapeGroup` has no single Coal collision geometry of
    its own -- it is never itself passed to Coal, only iterated -- so
    :meth:`to_dict` (there is no single dict that could describe a
    collection) and :meth:`_init_coal` (there is no single geometry to
    build) both raise if called directly. Nothing in normal use calls
    either, since ``iscollided()``/``closest_point()`` never touch
    ``self.co``.

    Every element's ``scene_parent`` is kept pointing at the group, same as
    :class:`SceneGroup` -- adding an element (via the constructor,
    ``append``, ``extend``, ``insert``, or item assignment) sets it, and
    removing one (``remove``, ``pop``, ``clear``, ``del``) clears it back
    to ``None``.

    .. runblock:: pycon

        >>> from spatialgeometry import CollisionShapeGroup, Cuboid, Sphere
        >>> from spatialmath import SE3
        >>> group = CollisionShapeGroup()
        >>> group.append(Cuboid([1, 1, 1]))
        >>> group.append(Sphere(1, pose=SE3.Trans(3, 0, 0)))
        >>> len(group)
    """

    def __init__(
        self,
        initlist: Iterable[CollisionShape] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            scene_children=(
                [self._check_type(item) for item in initlist]
                if initlist is not None
                else None
            ),
            **kwargs,
        )

    @staticmethod
    def _check_type(item: object) -> CollisionShape:
        if not isinstance(item, (CollisionShape, CollisionShapeGroup)):
            raise TypeError(
                "CollisionShapeGroup only accepts CollisionShape or "
                f"CollisionShapeGroup instances, got {type(item).__name__}"
            )
        return item

    def __getitem__(self, i: int) -> CollisionShape:
        return self._scene_children[i]

    def __setitem__(self, i: int, item: CollisionShape) -> None:
        if isinstance(i, slice):
            raise NotImplementedError(
                "Slice assignment is not supported for CollisionShapeGroup"
            )
        self._check_type(item)
        old = self._scene_children[i]
        self.remove(old)
        self.insert(i, item)

    def __delitem__(self, i: int | slice) -> None:
        if isinstance(i, slice):
            for item in list(self._scene_children[i]):
                self.remove(item)
        else:
            self.remove(self._scene_children[i])

    def append(self, item: CollisionShape) -> None:
        """Add ``item`` to the end of the group and set its
        ``scene_parent`` to this group."""
        self._check_type(item)
        item.scene_parent = self

    def extend(self, other: Iterable[CollisionShape]) -> None:
        """Add every element of ``other`` to the end of the group, setting
        each one's ``scene_parent`` to this group."""
        for item in other:
            self.append(item)

    def insert(self, i: int, item: CollisionShape) -> None:
        """Insert ``item`` at position ``i`` and set its ``scene_parent``
        to this group."""
        self._check_type(item)
        item.scene_parent = self
        self._scene_children.insert(i, self._scene_children.pop())

    def remove(self, item: CollisionShape) -> None:
        """Remove ``item`` from the group and clear its ``scene_parent``."""
        self._remove_scene_child(item)
        item._scene_parent = None

    def pop(self, i: int = -1) -> CollisionShape:
        """Remove and return the element at position ``i`` (default: the
        last one), clearing its ``scene_parent``."""
        item = self._scene_children[i]
        self.remove(item)
        return item

    def clear(self) -> None:
        """Remove every element from the group, clearing each one's
        ``scene_parent``."""
        for item in list(self._scene_children):
            self.remove(item)

    @property
    def data(self) -> list[CollisionShape]:
        return self._scene_children

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.data!r})"

    # --------------------------------------------------------------------- #
    # Collision checking -- iterate this group's own elements, never touch
    # self.co (there isn't one).
    # --------------------------------------------------------------------- #

    def _init_coal(self) -> None:
        raise NotImplementedError(
            "CollisionShapeGroup has no single Coal collision geometry -- "
            "iscollided()/closest_point() iterate its elements instead, "
            "this should never be called directly"
        )

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(
            "CollisionShapeGroup cannot be represented as a single dict -- "
            "iterate it and call to_dict() on each element instead"
        )

    def closest_point(
        self, shape: CollisionShape, inf_dist: float = 1.0
    ) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        """
        Return the minimum euclidean distance between this group and
        shape -- the closest ``(d, p1, p2)`` among this group's own
        elements.

        :param shape: The shape (or :class:`CollisionShapeGroup`) to
            compare distance to
        :param inf_dist: Only return a result when distance < inf_dist
        :returns: (d, p1, p2) — distance and closest points in world frame,
            or (None, None, None) when every element is farther than
            inf_dist.
        """
        d = None
        p1 = None
        p2 = None

        for element in self:
            td, tp1, tp2 = element.closest_point(shape, inf_dist)
            if td is not None and (d is None or td < d):
                d = td
                p1 = tp1
                p2 = tp2

        return d, p1, p2

    def iscollided(self, shape: CollisionShape) -> bool:
        """
        Return True if any element of this group has collided with shape.

        :param shape: The shape (or :class:`CollisionShapeGroup`) to check
            against
        """
        return any(element.iscollided(shape) for element in self)


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
    :param color: Flat colour override, applied to every face/vertex. If
        not given, the renderer uses whatever per-vertex/per-face colours
        are baked into the mesh file itself, when present.
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
        color: ArrayLike | None = None,
        y_up: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(stype="mesh", color=color, **kwargs)
        self.filename = filename
        self.scale = scale
        self._use_vertex_colors = color is None
        self.y_up = y_up

    # Overrides Shape.color's setter (keeping its getter) purely to track
    # whether a caller has explicitly asked for a flat colour at any point
    # after construction too, not just via __init__'s color= above.
    @Shape.color.setter
    def color(self, value: ArrayLike) -> None:
        Shape.color.fset(self, value)
        self._use_vertex_colors = False

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
    @mark_changed
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
    @mark_changed
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
    @mark_changed
    def y_up(self, value: bool) -> None:
        self._y_up = bool(value)

    def to_dict(self) -> dict[str, Any]:
        shape = super().to_dict()
        shape["filename"] = self.filename
        shape["scale"] = self.scale.tolist()
        shape["use_vertex_colors"] = self._use_vertex_colors
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
                "bounding box. Install with:  pip install spatialgeometry[mesh]"
                " (or just  pip install trimesh  directly)"
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
        # Coal Cylinder(radius, length) takes full length, not half length
        # -- verified directly (coal.Cylinder(1.0, 10.0).halfLength == 5.0).
        # Passing length/2.0 here (as this line used to) silently built a
        # collision cylinder half as tall as gm.Cylinder's own documented
        # "length: Total length in metres" contract promises.
        geom = _coal.Cylinder(self.radius, self.length)
        self.co = _coal.CollisionObject(geom)
        self._cinit = True

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    @mark_changed
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    @mark_changed
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
    @mark_changed
    def radius(self, value: float) -> None:
        self._radius = float(value)

    def to_dict(self) -> dict[str, Any]:
        shape = super().to_dict()
        shape["radius"] = self.radius
        return shape

    def _local_corners(self) -> np.ndarray:
        r = self.radius
        return aabb_corners([-r, -r, -r], [r, r, r])


class Ellipsoid(CollisionShape):
    """
    An ellipsoid whose centre is at the local origin.

    :param radii: Semi-axis lengths along X, Y, Z, in metres. A sphere is
        the special case radii=[r, r, r] -- use :class:`Sphere` for that,
        it's cheaper both to construct and to collision-check.
    :param collision: Whether this shape participates in collision checking.
    """

    _repr_params = ("radii",)

    def __init__(self, radii: ArrayLike, **kwargs) -> None:
        super().__init__(stype="ellipsoid", **kwargs)
        self.radii = radii

    def _init_coal(self) -> None:
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        r = self.radii
        self.co = _coal.CollisionObject(_coal.Ellipsoid(r[0], r[1], r[2]))
        self._cinit = True

    @property
    def radii(self) -> np.ndarray:
        return self._radii

    @radii.setter
    @mark_changed
    def radii(self, value: ArrayLike) -> None:
        value = getvector(value, 3)
        self._radii = np.array(value)

    def to_dict(self) -> dict[str, Any]:
        shape = super().to_dict()
        shape["radii"] = self.radii.tolist()
        return shape


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
    @mark_changed
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
