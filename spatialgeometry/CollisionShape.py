#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

import numpy as np
from io import StringIO
from spatialmath.base import r2q
from spatialmath.base.argcheck import getvector
from functools import wraps
from spatialmath import SE3
from spatialgeometry import Shape
import os

p = None
_pyb = None


def _import_pyb():
    import importlib
    global _pyb
    global p

    try:
        from spatialgeometry.tools.stdout_supress import pipes
    except Exception:  # pragma nocover
        from contextlib import contextmanager

        @contextmanager
        def pipes(stdout=None, stderr=None):
            pass

    try:
        out = StringIO()
        try:
            with pipes(stdout=out, stderr=None):
                p = importlib.import_module('pybullet')
        except Exception:  # pragma nocover
            p = importlib.import_module('pybullet')

        cid = p.connect(p.SHARED_MEMORY)
        if (cid < 0):
            p.connect(p.DIRECT)
        _pyb = True
    except ImportError:   # pragma nocover
        _pyb = False


class CollisionShape(Shape):

    def __init__(self, collision=True, **kwargs):
        self._collision = collision
        self.co = None
        self.pinit = False
        super().__init__(**kwargs)

    def _update_pyb(self):
        if _pyb and self.co is not None:
            q = r2q(self._sT[:3, :3])
            rot = [q[1], q[2], q[3], q[0]]
            p.resetBasePositionAndOrientation(
                self.co, self._sT[:3, 3], rot)

    def _init_pob(self):   # pragma nocover
        pass

    def _check_pyb(func):   # pragma nocover
        @wraps(func)
        def wrapper_check_pyb(*args, **kwargs):
            if _pyb is None:
                _import_pyb()
            return func(*args, **kwargs)
        return wrapper_check_pyb

    def _init_pob(self, col):
        self.co = p.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=col)
        self.pinit = True

    @property
    def wT(self):
        return self._wT @ self.base.A

    @wT.setter
    def wT(self, T):
        self._wT = T
        self._sT = self._wT @ self._base.A
        self._update_pyb()

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._base = T
        self._sT = self._wT @ self._base.A
        self._update_pyb()

    @property
    def collision(self):
        return self._collision

    @_check_pyb
    def closest_point(self, shape, inf_dist=1.0):
        '''
        closest_point(shape, inf_dist) returns the minimum euclidean
        distance between self and shape, provided it is less than inf_dist.
        It will also return the points on self and shape in the world frame
        which connect the line of length distance between the shapes. If the
        distance is negative then the shapes are collided.

        :param shape: The shape to compare distance to
        :type shape: Shape
        :param inf_dist: The minimum distance within which to consider
            the shape
        :type inf_dist: float
        :returns: d, p1, p2 where d is the distance between the shapes,
            p1 and p2 are the points in the world frame on the respective
            shapes
        :rtype: float, SE3, SE3
        '''

        if not self.pinit:
            self._init_pob()
            self._update_pyb()

        if not shape.pinit:
            shape._init_pob()
            shape._update_pyb()

        if not _pyb:  # pragma nocover
            raise ImportError(
                'The package PyBullet is required for collision '
                'functionality. Install using pip install pybullet')

        ret = p.getClosestPoints(self.co, shape.co, inf_dist)

        if len(ret) == 0:
            d = None
            p1 = None
            p2 = None
        else:
            ret = ret[0]
            d = ret[8]
            p1 = SE3(ret[5])
            p2 = SE3(ret[6])

        return d, p1, p2

    def collided(self, shape):
        '''
        collided(shape) checks if self and shape have collided

        :param shape: The shape to compare distance to
        :type shape: Shape
        :returns: True if shapes have collided
        :rtype: bool
        '''

        d, _, _ = self.closest_point(shape)

        if d is not None and d <= 0:
            return True
        else:
            return False


class Mesh(CollisionShape):
    """
    A mesh object described by an stl or collada file.

    :param filename: The path to the mesh that contains this object.
        This is the absolute path.
    :type filename: str
    :param scale: The scaling value for the mesh along the XYZ axes. If
        ``None``, assumes no scale is applied.
    :type scale: list (3) float, optional
    :param base: Local reference frame of the shape
    :type base: SE3
    :param collision: This shape is being used as a collision object
    :type collision: bool

    """

    def __init__(self, filename=None, scale=[1, 1, 1], **kwargs):
        super(Mesh, self).__init__(stype='mesh', **kwargs)

        self.filename = filename
        self.scale = scale

    def _init_pob(self):
        name, file_extension = os.path.splitext(self.filename)
        if (file_extension == '.stl' or file_extension == '.STL') and self.collision:

            col = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=self.filename,
                meshScale=self.scale)

            super()._init_pob(col)
        else:
            raise ValueError(
                "This shape has self.collision=False meaning it "
                "is not to be used as a collision object")

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = getvector(value, 3)
        else:
            value = getvector([1, 1, 1], 3)
        self._scale = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    def to_dict(self):
        '''
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        '''

        shape = super().to_dict()
        shape['filename'] = self.filename
        shape['scale'] = self.scale.tolist()
        return shape


class Cylinder(CollisionShape):
    """A cylinder whose center is at the local origin.
    Parameters

    :param radius: The radius of the cylinder in meters.
    :type radius: float
    :param length: The length of the cylinder in meters.
    :type length: float
    :param base: Local reference frame of the shape
    :type base: SE3
    :param collision: This shape is being used as a collision object
    :type collision: bool

    """

    def __init__(self, radius, length, **kwargs):
        super(Cylinder, self).__init__(stype='cylinder', **kwargs)
        self.radius = radius
        self.length = length

    def _init_pob(self):
        if self.collision:
            col = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.radius, height=self.length)

            super()._init_pob(col)
            super()._init_pob(col)
        else:
            raise ValueError(
                "This shape has self.collision=False meaning it "
                "is not to be used as a collision object")

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)

    def to_dict(self):
        '''
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        '''

        shape = super().to_dict()
        shape['radius'] = self.radius
        shape['length'] = self.length
        return shape


class Sphere(CollisionShape):
    """
    A sphere whose center is at the local origin.

    :param radius: The radius of the sphere in meters.
    :type radius: float
    :param base: Local reference frame of the shape
    :type base: SE3
    :param collision: This shape is being used as a collision object
    :type collision: bool

    """

    def __init__(self, radius, **kwargs):
        super(Sphere, self).__init__(stype='sphere', **kwargs)
        self.radius = radius

    def _init_pob(self):
        if self.collision:
            col = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE, radius=self.radius)

            super()._init_pob(col)
        else:
            raise ValueError(
                "This shape has self.collision=False meaning it "
                "is not to be used as a collision object")

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)

    def to_dict(self):
        '''
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        '''

        shape = super().to_dict()
        shape['radius'] = self.radius
        return shape


class Box(CollisionShape):
    """
    A rectangular prism whose center is at the local origin.

    :param scale: The length, width, and height of the box in meters.
    :type scale: list (3) float
    :param base: Local reference frame of the shape
    :type base: SE3
    :param collision: This shape is being used as a collision object
    :type collision: bool

    """

    def __init__(self, scale, **kwargs):
        super(Box, self).__init__(stype='box', **kwargs)
        self.scale = scale

    def _init_pob(self):

        if self.collision:
            col = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=np.array(self.scale)/2)

            super()._init_pob(col)
        else:
            raise ValueError(
                "This shape has self.collision=False meaning it "
                "is not to be used as a collision object")

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = getvector(value, 3)
        else:
            value = getvector([1, 1, 1], 3)
        self._scale = value

    def to_dict(self):
        '''
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        '''

        shape = super().to_dict()
        shape['scale'] = self.scale.tolist()
        return shape
