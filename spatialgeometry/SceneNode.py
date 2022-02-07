#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from numpy import ndarray, eye, zeros, copy as npcopy
from spatialmath.base import r2q
from abc import ABC

# from roboticstoolbox.robot.ETS import ETS
from typing import Type


class SceneNode(ABC):
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

        # The quaternion extracted from wT
        self.__wq = zeros(4)

        # The local transform
        self.__T = T

        self.__scene_children = children

        # Update child parents
        for child in children:
            child._scene_parent = self

        self.__scene_parent = parent

    # TODO DEFINE COPY METHOD

    @property
    def _scene_parent(self) -> Type["SceneNode"]:
        """
        Returns the parent node of this object

        """
        return self.__scene_parent

    @property
    def _scene_children(self) -> list["SceneNode"]:
        """
        Returns the child nodes of this object

        """
        return self.__scene_children

    @property
    def _wT(self) -> ndarray:
        """
        Returns the transform of this object in the world frame

        """
        return self.__wT

    @property
    def _wq(self) -> ndarray:
        """
        Returns the quaternion of this object in the world frame.

        """
        return self.__wq

    @property
    def _T(self) -> ndarray:
        """
        Returns the transform of this object with respect to the parent
        frame.

        """
        return npcopy(self.__T)

    @_T.setter
    def _T(self, T: ndarray):
        self.__T[:] = T

        if self.__scene_parent is not None:
            self.__wT[:] = self.parent.wT @ self._T
        else:
            self.__wT[:] = self._T

        self.__wq[:] = r2q(self.__wT[:3, :3], order="xyzs")

    # Attach should go here
