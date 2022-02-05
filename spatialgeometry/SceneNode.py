#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from numpy import ndarray, eye, zeros
from spatialmath.base import r2q
from roboticstoolbox.robot.ETS import ETS


class SceneNode:
    def __init__(
        self,
        T: ndarray = eye(4),
        parent: "SceneNode" = None,
        children: list["SceneNode"] = [],
    ):

        # These three are static attributes which can never be changed
        # If these are directly accessed and re-written, segmentation faults
        # will follow very soon after
        # wT and sT cannot be accessed and set by users by base can be
        # modified through its setter

        # The world transform
        self.__wT = eye(4)

        # The swift transform, may have a constant offset from wT
        self.__swift_wT = eye(4)

        # The swift quaternion extracted from swift_wT
        self.__swift_wq = zeros(4)

        # The local transform
        self.__T = T

        self._children = children
        self._parent = parent

    @property
    def parent(self) -> "SceneNode":
        """
        Returns the parent node of this object

        """
        return self._parent

    @property
    def children(self) -> list["SceneNode"]:
        """
        Returns the child nodes of this object

        """
        return self._children

    @property
    def wT(self) -> ndarray:
        """
        Returns the transform of this object in the world frame

        """
        return self.__wT

    @property
    def swift_wT(self) -> ndarray:
        """
        Returns the transform of this object in the world frame using
        the three.js representation of the object (if applicable).

        Note that this property is only useful to swift not end-users.

        """
        return self.__swift_wT

    @property
    def swift_wq(self) -> ndarray:
        """
        Returns the quaternion of this object in the world frame using
        the three.js representation of the object (if applicable).

        Note that this property is only useful to swift not end-users.

        """
        return self.__swift_wq

    @property
    def T(self) -> ndarray:
        """
        Returns the transform of this object with respect to the parent
        frame.

        """
        return self.__T

    @T.setter
    def T(self, T: ndarray):
        self.__T[:] = T

        if self.parent is not None:
            self.__wT[:] = self.parent.wT @ self.T
        else:
            self.__wT[:] = self.T

        self.__swift_wq[:] = r2q(self.__wT[:3, :3], order="xyzs")
