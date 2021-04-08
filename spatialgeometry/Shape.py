#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from spatialmath.base import r2q
import numpy as np

_mpl = False

try:
    from matplotlib import colors as mpc
    _mpl = True
except ImportError:    # pragma nocover
    pass


CONST_RX = SE3.Rx(np.pi / 2).A


class Shape:

    def __init__(
            self,
            base=None,
            color=None,
            stype=None):

        # These three are static attributes which can never be changed
        # If these are directly accessed and re-written, segmentation faults
        # will follow very soon after
        # wT and sT cannot be accessed and set by users by base can be
        # modified through its setter
        self._wT = np.eye(4)
        self._sT = np.eye(4)
        self._base = np.eye(4)

        self.base = base
        self.stype = stype
        self.v = np.zeros(6)
        self.color = color

    def _to_hex(self, rgb):
        rgb = (np.array(rgb) * 255).astype(int)
        return int('0x%02x%02x%02x' % (rgb[0], rgb[1], rgb[2]), 16)

    def to_dict(self):
        '''
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        '''
        self._to_hex(self.color[0:3])

        if self.stype == 'cylinder':
            fk = self._sT @ CONST_RX
        else:
            fk = self._sT

        q = r2q(fk[:3, :3]).tolist()
        q = [q[1], q[2], q[3], q[0]]

        shape = {
            'stype': self.stype,
            't': fk[:3, 3].tolist(),
            'q': q,
            'v': self.v.tolist(),
            'color': self._to_hex(self.color[0:3]),
            'opacity': self.color[3]
        }

        return shape

    def fk_dict(self):
        '''
        fk_dict() outputs shapes pose in dictionary form

        :returns: The shape pose in translation and quternion form
        :rtype: dict
        '''

        if self.stype == 'cylinder':
            fk = self._sT @ CONST_RX
        else:
            fk = self._sT

        q = r2q(fk[:3, :3]).tolist()
        q = [q[1], q[2], q[3], q[0]]

        shape = {
            't': fk[:3, 3].tolist(),
            'q': q
        }

        return shape

    def __repr__(self):   # pragma nocover
        return f'{self.stype},\n{self.base}'

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = getvector(value, 6)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):

        default_color = (0.95, 0.5, 0.25, 1.0)

        if isinstance(value, str):
            if _mpl:
                try:
                    value = mpc.to_rgba(value)
                except ValueError:
                    print(
                        f'{value} is an invalid color '
                        'name, using default color')
                    value = default_color
            else:  # pragma nocover
                value = default_color
                print(
                    'Color only supported when matplotlib is installed\n'
                    'Install using: pip install matplotlib')
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

    @property
    def wT(self):
        return self._wT @ self.base.A

    @wT.setter
    def wT(self, T):
        self._wT[:] = T
        self._sT[:] = self._wT @ self._base

    @property
    def base(self):
        return SE3(np.copy(self._base), check=False)

    @base.setter
    def base(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._base[:] = T.A
        self._sT[:] = self._wT @ self._base
