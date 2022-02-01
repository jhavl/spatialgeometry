#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from spatialmath.base import r2q
import numpy as np
import copy

_mpl = False
_rtb = False

try:
    from matplotlib import colors as mpc

    _mpl = True
except ImportError:  # pragma nocover
    pass



try:
    import roboticsroolbox as rtb

    _rtb = True
except ImportError:  # pragma nocover
    pass


CONST_RX = SE3.Rx(np.pi / 2).A


class Shape:
    def __init__(self, base=None, color=None, stype=None):

        # These three are static attributes which can never be changed
        # If these are directly accessed and re-written, segmentation faults
        # will follow very soon after
        # wT and sT cannot be accessed and set by users by base can be
        # modified through its setter

        # The world transform
        self._wT = np.eye(4)

        # The swift transform, may have a constant offset from wT
        self._sT = np.eye(4)

        # The swift quaternion extracted from sT
        self._sq = np.zeros(4)

        # The 
        self._base = np.eye(4)

        self.base = base
        self.stype = stype
        self.v = np.zeros(6)
        self.color = color
        self.attached = True

        self._collision = False

    def attach_to(self, object):
        if isinstance(object, Shape):
            self.attached = True
            self._wT = object._wT



    def copy(self):
        """
        Copy of Shape object

        :return: Shallow copy of Shape object
        :rtype: Shape
        """

        # print("Hello")
        new = copy.copy(self)

        # print(self._base)

        # new = Shape(self.base, self.color, self.stype)

        for k, v in self.__dict__.items():
            if k.startswith("_") and isinstance(v, np.ndarray):
                setattr(new, k, np.copy(v))

        return new

    def _to_hex(self, rgb):
        rgb = (np.array(rgb) * 255).astype(int)
        return int("0x%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]), 16)

    def to_dict(self):
        """
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        """
        self._to_hex(self.color[0:3])

        if self.stype == "cylinder":
            fk = self._sT @ CONST_RX
        else:
            fk = self._sT

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

    def fk_dict(self):
        """
        fk_dict() outputs shapes pose in dictionary form

        :returns: The shape pose in translation and quternion form
        :rtype: dict
        """

        if self.stype == "cylinder":
            fk = self._sT @ CONST_RX
        else:
            fk = self._sT

        q = r2q(fk[:3, :3]).tolist()
        q = [q[1], q[2], q[3], q[0]]

        shape = {"t": fk[:3, 3].tolist(), "q": q}

        return shape

    def __repr__(self):  # pragma nocover
        return f"{self.stype},\n{self.base}"

    @property
    def collision(self):
        return self._collision

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = getvector(value, 6)

    @property
    def color(self):
        """
        shape.color returns a four length tuple representing (red, green, blue, alpha)
        where alpha represents transparency. Values returned are in the range [0-1].
        """
        return self._color

    @color.setter
    def color(self, value):
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

            value = np.array(value)

            if np.any(value > 1.0):
                value = value / 255.0

            if value.shape[0] == 3:
                value = np.r_[value, 1.0]

            value = tuple(value)

        self._color = value

    def set_alpha(self, alpha):
        """
        Convenience method to set the opacity/alpha value of the robots color.
        """

        if alpha > 1.0:
            alpha /= 255

        new_color = np.r_[self._color[:3], alpha]
        self._color = tuple(new_color)

    @property
    def wT(self):
        return self._sT

    @wT.setter
    def wT(self, T):
        self._wT[:] = T
        self._sT[:] = self._wT @ self._base
        self._sq[:] = r2q(self._sT[:3, :3], order="xyzs")

    @property
    def base(self):
        return SE3(np.copy(self._base), check=False)

    @base.setter
    def base(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._base[:] = T.A
        self._sT[:] = self._wT @ self._base
        self._sq[:] = r2q(self._sT[:3, :3], order="xyzs")


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
