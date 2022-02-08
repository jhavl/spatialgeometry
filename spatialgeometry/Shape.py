#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from spatialgeometry.SceneNode import SceneNode
from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from spatialmath.base import r2q
from copy import copy as ccopy
from numpy import ndarray, copy as npcopy, pi, zeros, array, any, concatenate, eye
from typing import Union

ArrayLike = Union[list, ndarray, tuple, set]
_mpl = False
_rtb = False

try:
    from matplotlib import colors as mpc

    _mpl = True
except ImportError:  # pragma nocover
    pass


try:
    import roboticstoolbox as rtb

    _rtb = True
except ImportError:  # pragma nocover
    pass


CONST_RX = SE3.Rx(pi / 2).A


class Shape(SceneNode):
    def __init__(
        self,
        T: Union[ndarray, SE3] = eye(4),
        color: ArrayLike = None,
        stype: str = None,
        **kwargs,
    ):

        # Initialise the scene node
        super().__init__(T=T, **kwargs)

        self.stype = stype
        self.v = zeros(6)
        self.color = color
        self.attached = True

        self._collision = False

    def attach_to(self, object):
        if isinstance(object, Shape):
            self.attached = True
            self._wT = object._wT

    def copy(self) -> "Shape":
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

    def _to_hex(self, rgb) -> int:
        rgb = (array(rgb) * 255).astype(int)
        return int("0x%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]), 16)

    def to_dict(self) -> str:
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """
        self._to_hex(self.color[0:3])

        if self.stype == "cylinder":
            fk = self._wT @ CONST_RX
        else:
            fk = self._wT

        q = r2q(fk[:3, :3]).tolist()
        q = [q[1], q[2], q[3], q[0]]

        shape = {
            "stype": self.stype,
            "t": fk[:3, 3].tolist(),
            "q": q,
            "v": self.v.tolist(),
            "color": self._to_hex(self.color[0:3]),
            "opacity": self.color[3],
        }

        return shape

    def fk_dict(self) -> str:
        """
        fk_dict() outputs shapes pose in dictionary form

        :returns: The shape pose in translation and quternion form
        :rtype: dict
        """

        if self.stype == "cylinder":
            fk = self._wT @ CONST_RX
        else:
            fk = self._wT

        q = r2q(fk[:3, :3]).tolist()
        q = [q[1], q[2], q[3], q[0]]

        shape = {"t": fk[:3, 3].tolist(), "q": q}

        return shape

    def __repr__(self) -> str:  # pragma nocover
        return f"{self.stype},\n{self.T[:3, -1]}"

    @property
    def collision(self) -> bool:
        return self._collision

    @property
    def v(self) -> ndarray:
        return self._v

    @v.setter
    def v(self, value: ArrayLike):
        self._v = getvector(value, 6)

    @property
    def color(self) -> list[float]:
        """
        shape.color returns a four length tuple representing (red, green, blue, alpha)
        where alpha represents transparency. Values returned are in the range [0-1].
        """
        return self._color

    @color.setter
    def color(self, value: ArrayLike):
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

            if value.shape[0] == 3:
                value = concatenate([value, [1.0]])

            value = tuple(value)

        self._color = value

    def set_alpha(self, alpha: Union[float, int]):
        """
        Convenience method to set the opacity/alpha value of the robots color.
        """

        if alpha > 1.0:
            alpha /= 255

        new_color = concatenate([self._color[:3], [alpha]])
        self._color = tuple(new_color)

    # --------------------------------------------------------------------- #
    # SceneNode properties
    # These relate to how scene node operates

    @property
    def T(self) -> ndarray:
        return self._T

    @T.setter
    def T(self, T_new: Union[ndarray, SE3]):
        if isinstance(T_new, SE3):
            T_new = T_new.A
        self._T = T_new

    # --------------------------------------------------------------------- #


class Axes(Shape):
    """An axes whose center is at the local origin.
    Parameters

    :param length: The length of each axis.
    :type length: float
    :param base: Local reference frame of the shape
    :type base: SE3

    """

    def __init__(self, length, **kwargs):
        super(Axes, self).__init__(stype="axes", **kwargs)
        self.length = length

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)

    def to_dict(self):
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """

        shape = super().to_dict()
        shape["length"] = self.length
        return shape


class Arrow(Shape):
    """An arrow whose center is at the local origin, and points
    in the positive z direction.

    Parameters

    :param length: The length of the arrow.
    :type length: float
    :param base: Local reference frame of the shape
    :type base: SE3

    """

    def __init__(self, length, **kwargs):
        super(Arrow, self).__init__(stype="arrow", **kwargs)
        self.length = length

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)

    def to_dict(self):
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """

        shape = super().to_dict()
        shape["length"] = self.length
        return shape
