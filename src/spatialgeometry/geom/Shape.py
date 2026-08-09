#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from spatialgeometry.geom.SceneNode import SceneNode
from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from copy import copy as ccopy, deepcopy
from numpy import (
    ndarray,
    copy as npcopy,
    pi,
    zeros,
    array,
    any,
    concatenate,
    eye,
    array_equal,
)
from typing import Any
from warnings import warn

import numpy as np

ArrayLike = list | ndarray | tuple | set
_mpl = False


def aabb_corners(mn: ArrayLike, mx: ArrayLike) -> ndarray:
    """
    The 8 corners of the axis-aligned box spanning ``mn`` to ``mx``.

    :param mn: minimum [x, y, z]
    :param mx: maximum [x, y, z]
    :rtype: ndarray(3,8)
    """
    return np.array(
        [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    ).T
# _rtb = False


def update(func):  # pragma nocover
    @wraps(func)
    def wrapper_update(*args, **kwargs):

        if args[0]._added_to_swift:
            args[0]._changed = True

        return func(*args, **kwargs)

    return wrapper_update


try:
    from matplotlib import colors as mpc

    _mpl = True
except ImportError:  # pragma nocover
    pass


# try:
#     import roboticstoolbox as rtb

#     _rtb = True
# except ImportError:  # pragma nocover
#     pass


CONST_RX = SE3.Rx(pi / 2).A


class Shape(SceneNode, ABC):
    """
    Abstract base class for a single renderable/collidable object in the
    scene (a primitive, a mesh, or a Path). Not instantiated directly --
    see the concrete subclasses in this module and in
    :class:`~spatialgeometry.geom.CollisionShape.CollisionShape`.
    """

    #: Names of this class's own constructor arguments to include in
    #: :meth:`__repr__`, in the order they should appear.
    _repr_params: tuple[str, ...] = ()

    def __init__(
        self,
        pose: ndarray | SE3 = eye(4),
        color: ArrayLike | None = None,
        stype: str | None = None,
        base: ndarray | SE3 | None = None,
        **kwargs,
    ) -> None:
        """
        :param pose: Local reference frame of the shape, defaults to the
            identity transform.
        :param color: Colour as (r, g, b) or (r, g, b, a) in [0-1] (or
            [0-255], auto-normalised), or a matplotlib colour name. Defaults
            to a mid-grey ``(0.3, 0.3, 0.3, 1.0)``.
        :param stype: Shape type identifier used by the renderer/wire
            protocol (e.g. ``"cuboid"``, ``"mesh"``) -- set by each concrete
            subclass, not normally passed directly by a caller.
        :param base: Deprecated alias for ``pose``.
        """

        # Swift related attributes
        self._added_to_swift = False
        self._changed = False

        if base is not None:
            warn("base kwarg is deprecated, use pose instead", FutureWarning)

            if isinstance(base, SE3):
                T = base.A
            else:
                T = base

            if T is not None and not array_equal(pose, eye(4)):
                raise ValueError(
                    "You cannot use both base and pose kwargs as they offer identical functionality. Use only pose."
                )

        else:

            if isinstance(pose, SE3):
                T = pose.A
            else:
                T = pose

        if color is None:
            self._color = (0.3, 0.3, 0.3, 1.0)
        else:
            self.color = color

        # Initialise the scene node
        super().__init__(T=T, **kwargs)

        self.stype = stype
        self.v = zeros(6)
        self.attached = True

        self._collision = False

    # --------------------------------------------------------------------- #

    def copy(self) -> Shape:
        """
        Copy of Shape object

        :return: Shallow copy of Shape object
        :rtype: Shape
        """

        new = ccopy(self)

        for k, v in self.__dict__.items():
            if k.startswith("_") and isinstance(v, ndarray):
                setattr(new, k, npcopy(v))

        return new

    def __copy__(self):
        return deepcopy(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if not k.lower().startswith("_scene"):
                setattr(result, k, deepcopy(v, memo))

        result._custom_scene_node_init(T=deepcopy(self.T))

        return result

    # --------------------------------------------------------------------- #

    def _to_hex(self, rgb: ArrayLike) -> int:
        rgb = (array(rgb) * 255).astype(int)
        return int("0x%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]), 16)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """
        self._to_hex(self.color[0:3])

        shape = {
            "stype": self.stype,
            "t": self._wT[:3, 3].tolist(),
            "q": self._wq.tolist(),
            "v": self.v.tolist(),
            "color": self._to_hex(self.color[0:3]),
            "opacity": self.color[3],
        }

        return shape

    def fk_dict(self) -> dict[str, Any]:
        """
        fk_dict() outputs shapes pose in dictionary form

        :returns: The shape pose in translation and quternion form
        :rtype: dict
        """

        # q = smb.r2q(self._wT[:3, :3])
        # q = [q[1], q[2], q[3], q[0]]
        # shape = {"t": self._wT[:3, 3].tolist(), "q": q}

        shape = {"t": self._wT[:3, 3].tolist(), "q": self._wq.tolist()}

        return shape

    def __repr__(self) -> str:
        args = []
        for name in self._repr_params:
            value = getattr(self, name)
            if isinstance(value, ndarray):
                value = value.tolist()
            args.append(f"{name}={value!r}")

        # float(...) here, not just tuple(self.color[:3]) -- self._color's
        # elements are sometimes numpy.float64 (e.g. after color=[...] with
        # a list/array input), which reprs as "np.float64(1.0)" instead of
        # a plain "1.0". Harmless for JSON (float64 genuinely subclasses
        # float, unlike int64), but ugly here specifically.
        args.append(f"color={tuple(float(c) for c in self.color[:3])!r}")
        if self.color[3] != 1.0:
            args.append(f"opacity={float(self.color[3])!r}")

        args.append(f"pose={SE3(self._T, check=False).strline()!r}")
        return f"{type(self).__name__}({', '.join(args)})"

    @property
    def collision(self) -> bool:
        return self._collision

    @property
    def v(self) -> ndarray:
        return self._v

    @v.setter
    def v(self, value: ArrayLike) -> None:
        self._v = array(getvector(value, 6))

    @property
    def color(self) -> tuple[float, float, float, float]:
        """
        shape.color returns a four length tuple representing (red, green, blue, opacity)
        where opacity represents transparency. Values returned are in the range [0-1].
        See :attr:`opacity` for a convenient way to get/set just this last channel.
        """
        return self._color

    @color.setter
    @update
    def color(self, value: ArrayLike) -> None:
        """
        shape.color(new_color) sets the color of a shape.

        The color format is (red, green, blue, opacity).

        Color can be set with a three length list, tuple or array which
        will only set the (r, g, b) values and opacity will be set to maximum.

        Color can be set with a four length list, tuple or array which
        will set the (r, g, b, opacity) values.

        Note: the color is auto-normalising. If any value passed is greater than
        1.0 then all values will be normalised to the [0-1] range assuming the
        previous range was [0-255].
        """

        default_color = (0.95, 0.5, 0.25, 1.0)

        if isinstance(value, str):
            if _mpl:
                try:
                    value = mpc.to_rgba(value)
                except ValueError:
                    print(f"{value} is an invalid color name, using default color")
                    value = default_color
            else:  # pragma nocover
                value = default_color
                print(
                    "Color only supported when matplotlib is installed\n"
                    "Install using: pip install matplotlib"
                )
        elif value is None:
            value = default_color
        else:
            # dtype=float forced explicitly -- an all-integer input (e.g.
            # the very natural color=[1, 0, 0, 1] for opaque red) would
            # otherwise stay int64 whenever nothing needs 0-255
            # normalisation below, and self.color[3] (opacity) being a
            # numpy.int64 rather than a float breaks real (non-mocked)
            # json.dumps() sends in SwiftRoute.py -- TypeError: Object of
            # type int64 is not JSON serializable. Never caught by the
            # existing protocol tests since they're FakeBrowser-mocked and
            # never actually round-trip through json.dumps().
            value = array(value, dtype=float)

            if any(value > 1.0):
                value = value / 255.0

            if value.shape[0] == 3:  # type: ignore
                value = concatenate([value, [1.0]])

            value = tuple(value)

        self._color = value

    @property
    def opacity(self) -> float:
        """
        The last channel of :attr:`color`, in [0-1] -- 1.0 is fully
        opaque, 0.0 fully transparent. A convenience for touching just
        this channel without needing to know or re-specify the current
        (r, g, b).

        .. note::
            "Opacity" here is the same quantity commonly called "alpha"
            in computer graphics (as in RGBA) -- this package uses
            "opacity" consistently as the public name for it.

        This is a read/write property.

        :rtype: float
        """
        return self._color[3]

    @opacity.setter
    @update
    def opacity(self, value: float) -> None:
        if value > 1.0:
            value /= 255

        self._color = tuple(concatenate([self._color[:3], [value]]))

    def set_alpha(self, alpha: float | int) -> None:
        """Deprecated -- use the ``opacity`` property instead."""
        warn("set_alpha is deprecated, use the opacity property instead", FutureWarning)
        self.opacity = alpha

    # --------------------------------------------------------------------- #
    # Bounding box
    #
    # _local_corners() is the one thing each subclass overrides -- the 8
    # corners of the shape's own axis-aligned bounding box, in its local
    # frame (pose ignored). corners()/bounds()/extents() are all derived
    # from it here, generically, once. Not an @abstractmethod: a shape
    # that hasn't implemented it yet (e.g. Axes/Arrow/Path, currently)
    # raises NotImplementedError only if corners() is actually called on
    # it, rather than making the class impossible to instantiate.
    # --------------------------------------------------------------------- #

    def _local_corners(self) -> ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a bounding box"
        )

    def corners(self, world: bool = False) -> ndarray:
        """
        The 8 corners of this shape's axis-aligned bounding box.

        :param world: If True, apply this shape's current pose and return
            the axis-aligned envelope of the posed shape (its corners will
            move as the shape is re-posed, and a rotated shape's envelope
            is generally larger than its own local box -- this is *not*
            the shape's true oriented/rotated corners). If False
            (default), return the corners in the shape's own local frame,
            independent of pose -- constant unless the shape's own
            parameters (radius, scale, ...) change.

        :rtype: ndarray(3,8)
        """
        c = self._local_corners()
        if not world:
            return c

        wc = self._wT[:3, :3] @ c + self._wT[:3, 3:4]
        return aabb_corners(wc.min(axis=1), wc.max(axis=1))

    def bounds(self, world: bool = False) -> ndarray:
        """
        Min/max extent of this shape's axis-aligned bounding box along
        each axis.

        :param world: See :meth:`corners`.
        :rtype: ndarray(3,2)
        """
        c = self.corners(world=world)
        return np.column_stack([c.min(axis=1), c.max(axis=1)])

    def extents(self, world: bool = False) -> ndarray:
        """
        Dimensions (width, depth, height) of this shape's axis-aligned
        bounding box.

        :param world: See :meth:`corners`.
        :rtype: ndarray(3,)
        """
        b = self.bounds(world=world)
        return b[:, 1] - b[:, 0]

    # --------------------------------------------------------------------- #


class Axes(Shape):
    """An axes whose center is at the local origin.
    Parameters

    :param length: The length of each axis.
    :type length: float
    :param arrows: If True, render each axis as a colored Arrow (red/
        green/blue for X/Y/Z) instead of a plain line.
    :type arrows: bool
    :param radius: Shaft radius of each arrow. Only used when
        arrows=True; passed straight through to each constituent Arrow
        (see Arrow's own radius/linewidth docs -- they are mutually
        exclusive, radius > 0 takes precedence).
    :type radius: float
    :param linewidth: Shaft width in pixels, only used when arrows=True
        and radius == 0. Passed straight through to each constituent
        Arrow.
    :type linewidth: float
    :param pose: Local reference frame of the shape
    :type pose: SE3

    """

    _repr_params = ("length", "arrows", "radius", "linewidth")

    def __init__(
        self,
        length: float,
        arrows: bool = False,
        radius: float = 0.0,
        linewidth: float = 1.0,
        **kwargs,
    ) -> None:
        super(Axes, self).__init__(stype="axes", **kwargs)
        self.length = length
        self.arrows = arrows
        self.radius = radius
        self.linewidth = linewidth

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    @update
    def length(self, value: float) -> None:
        self._length = float(value)

    @property
    def arrows(self) -> bool:
        return self._arrows

    @arrows.setter
    @update
    def arrows(self, value: bool) -> None:
        self._arrows = bool(value)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    @update
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def linewidth(self) -> float:
        return self._linewidth

    @linewidth.setter
    @update
    def linewidth(self, value: float) -> None:
        self._linewidth = float(value)

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """

        shape = super().to_dict()
        shape["length"] = self.length
        shape["arrows"] = self.arrows
        shape["radius"] = self.radius
        shape["linewidth"] = self.linewidth
        return shape


class Arrow(Shape):
    """An arrow whose center is at the local origin, and points
    in the positive z direction.

    The arrow is made using a cylinder and a cone

    Parameters

    :param length: The total length of the arrow.
    :param radius: The radius of the arrow shaft. If radius is 0, the
        shaft is rendered as a line instead of a cylinder -- see
        linewidth. radius and linewidth are mutually exclusive: radius
        > 0 always takes precedence, and linewidth is ignored in that
        case (a real cylinder mesh has no notion of a pixel width).
    :param linewidth: Width of the shaft in pixels. Only used when
        radius == 0.
    :param head_length: The lenght of the cone (head of the arrow). This is
        represented as a fraction of the lenght. Must be a value between 0
        and 1.
    :param head_radius: The width of the cone (head of the arrow). This is
        represented as a fraction of the head_length.

    :param pose: Local reference frame of the shape
    :type pose: SE3

    """

    _repr_params = ("length", "radius", "linewidth", "head_length", "head_radius")

    def __init__(
        self,
        length: float,
        radius: float = 0.0,
        linewidth: float = 1.0,
        head_length: float = 0.2,
        head_radius: float = 0.2,
        **kwargs,
    ) -> None:
        if head_length > 1.0 or head_length < 0.0:
            raise ValueError("Head length must be a value between 0 and 1")

        super(Arrow, self).__init__(stype="arrow", **kwargs)
        self.length = length
        self.radius = radius
        self.linewidth = linewidth
        self.head_length = head_length
        self.head_radius = head_radius

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    @update
    def length(self, value: float) -> None:
        self._length = float(value)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    @update
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def linewidth(self) -> float:
        return self._linewidth

    @linewidth.setter
    @update
    def linewidth(self, value: float) -> None:
        self._linewidth = float(value)

    @property
    def head_length(self) -> float:
        return self._head_length

    @head_length.setter
    @update
    def head_length(self, value: float) -> None:
        self._head_length = float(value)

    @property
    def head_radius(self) -> float:
        return self._head_radius

    @head_radius.setter
    @update
    def head_radius(self, value: float) -> None:
        self._head_radius = float(value)

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """

        shape = super().to_dict()
        shape["length"] = self.length
        shape["radius"] = self.radius
        shape["linewidth"] = self.linewidth
        shape["head_length"] = self.head_length
        shape["head_radius"] = self.head_radius
        return shape


class Path(Shape):
    """A polyline through a sequence of waypoints -- straight segments
    joining consecutive points, not a smoothed curve -- for drawing
    paths and trajectories in the scene.

    :param points: waypoints defining the polyline
    :type points: ArrayLike
    :param radius: tube radius; if 0, rendered as a line instead of a
        tube -- see linewidth. radius and linewidth are mutually
        exclusive: radius > 0 always takes precedence, and linewidth is
        ignored in that case (a real tube mesh has no notion of a pixel
        width).
    :param linewidth: Width of the line in pixels. Only used when
        radius == 0.

    :param pose: Local reference frame of the shape
    :type pose: SE3
    """

    _repr_params = ("points", "radius", "linewidth")

    def __init__(
        self,
        points: ArrayLike,
        radius: float = 0.0,
        linewidth: float = 1.0,
        **kwargs,
    ) -> None:
        super(Path, self).__init__(stype="path", **kwargs)
        self.points = points
        self.radius = radius
        self.linewidth = linewidth

    @property
    def points(self) -> np.ndarray:
        """
        :rtype: ndarray(3,n)
        """
        return self._points

    @points.setter
    @update
    def points(self, value: ArrayLike) -> None:
        value = np.array(value, dtype=float)
        if value.ndim != 2 or value.shape[0] != 3:
            raise ValueError(
                f"points must be a 3xN array of waypoints, got shape {value.shape}"
            )
        if value.shape[1] < 2:
            raise ValueError("points must contain at least 2 waypoints")
        self._points = value

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    @update
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def linewidth(self) -> float:
        return self._linewidth

    @linewidth.setter
    @update
    def linewidth(self, value: float) -> None:
        self._linewidth = float(value)

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """

        shape = super().to_dict()
        # Wire format is a flat list of [x, y, z] waypoints (N x 3) -- the
        # natural shape for the JS side to consume directly (one
        # THREE.Vector3 per point) -- even though points is stored/accepted
        # here as 3xN, matching this ecosystem's own point-set convention.
        shape["points"] = self.points.T.tolist()
        shape["radius"] = self.radius
        shape["linewidth"] = self.linewidth
        return shape
