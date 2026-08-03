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
    def __init__(
        self,
        pose: ndarray | SE3 = eye(4),
        color: ArrayLike | None = None,
        stype: str | None = None,
        base: ndarray | SE3 | None = None,
        **kwargs,
    ) -> None:

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

    def __str__(self) -> str:
        return f"stype: {self.stype} \n pose: {SE3(self._T).t}"

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

    def __repr__(self) -> str:  # pragma nocover
        return f"{self.stype},\n{self.T[:3, -1]}"
        # return f"{hex(id(self.stype))}"

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
        shape.color returns a four length tuple representing (red, green, blue, alpha)
        where alpha represents transparency. Values returned are in the range [0-1].
        """
        return self._color

    @color.setter
    @update
    def color(self, value: ArrayLike) -> None:
        """
        shape.color(new_color) sets the color of a shape.

        The color format is (red, green, blue, alpha).

        Color can be set with a three length list, tuple or array which
        will only set the (r, g, b) values and alpha will be set to maximum.

        Color can be set with a four length list, tuple or array which
        will set the (r, g, b, a) values.

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
            value = array(value)

            if any(value > 1.0):
                value = value / 255.0

            if value.shape[0] == 3:  # type: ignore
                value = concatenate([value, [1.0]])

            value = tuple(value)

        self._color = value

    def set_alpha(self, alpha: float | int) -> None:
        """
        Convenience method to set the opacity/alpha value of the robots color.
        """

        if alpha > 1.0:
            alpha /= 255

        new_color = concatenate([self._color[:3], [alpha]])
        self._color = tuple(new_color)

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
