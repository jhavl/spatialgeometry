#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from numpy import ndarray, eye, zeros, copy as npcopy
from spatialmath.base import r2q
from abc import ABC
from scene import node_init, node_update, scene_graph_single
from spatialmath import SE3

# from roboticstoolbox.robot.ETS import ETS
from typing import Type, Union


class SceneNode:
    def __init__(
        self,
        T: ndarray = eye(4),
        scene_parent: Union["SceneNode", None] = None,
        scene_children: Union[list["SceneNode"], None] = None,
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
        self._T = eye(4)
        self._T[:] = T

        if scene_children is None:
            self._scene_children = []
        else:
            self._scene_children = scene_children

        self._scene_parent = scene_parent

        # Set up the c object
        self.__scene = self.__init_c()

        # Update childs parent
        for child in self.scene_children:
            child._update_scene_parent(self)

        # Update parents child
        if scene_parent is not None:
            scene_parent._update_scene_children(self)

    # --------------------------------------------------------------------- #

    def __init_c(self):
        """
        Super Private method which initialises a C object to hold Data

        """

        return node_init(
            len(self._scene_children),
            self._T,
            self.__wT,
            self.__wq,
            self._scene_parent._scene if self._scene_parent is not None else None,
            [child._scene for child in self._scene_children],
        )

    def __update_c(self):
        """
        Super Private method which updates the C object which holds Data

        """

        node_update(
            self.__scene,
            len(self._scene_children),
            self._scene_parent._scene if self._scene_parent is not None else None,
            [child._scene for child in self._scene_children],
        )

    @property
    def _scene(self):
        return self.__scene

    # --------------------------------------------------------------------- #

    # TODO DEFINE COPY METHOD

    def __str__(self) -> str:
        if self._scene_parent is not None:
            parent = f"{SE3(self._scene_parent._T, check=False).t}"
        else:
            parent = "None"

        # return f"parent: {parent} \n self: {SE3(self._T).t} \n children: {[SE3(child._T).t for child in self._scene_children]}"

        return f"parent: {parent} \n self: {SE3(self._T).t} \n children: {self._scene_children}"

    # --------------------------------------------------------------------- #

    @property
    def scene_parent(self) -> Type["SceneNode"]:
        """
        Returns the parent node of this object

        """
        return self._scene_parent

    @scene_parent.setter
    def scene_parent(self, parent: "SceneNode"):
        """
        Sets a new parent node of this object, will automatically update
        the parents child

        """
        # Set our parent
        self._scene_parent = parent

        # Update our parents children
        parent._update_scene_children(self)

        # Update c
        self.__update_c()

    def _update_scene_parent(self, parent: "SceneNode"):
        """
        Sets a new parent node of this object, does NOT update
        the parents child

        """
        self._scene_parent = parent

        # Update c
        self.__update_c()

    # --------------------------------------------------------------------- #

    @property
    def scene_children(self) -> list["SceneNode"]:
        """
        Returns the child nodes of this object

        """
        return self._scene_children

    @scene_children.setter
    def scene_children(self, children: list["SceneNode"]):
        """
        Sets the child nodes of this object, does not update childs
        parent

        """
        # Set our children
        self._scene_children = children

        # Update our childrens parent
        for child in children:
            child._update_scene_parent(self)

        # Update c
        self.__update_c()

    def _update_scene_children(self, child: "SceneNode"):
        """
        Appends a new child to this object, does NOT update
        the childs parent

        """
        self.scene_children.append(child)

        # Update c
        self.__update_c()

    # --------------------------------------------------------------------- #

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

    # @property
    # def _T(self) -> ndarray:
    #     """
    #     Returns the transform of this object with respect to the parent
    #     frame.

    #     """
    #     return npcopy(self.__T)

    # @_T.setter
    # def _T(self, T: ndarray):
    #     self.__T[:] = T

    #     if self.__scene_parent is not None:
    #         self.__wT[:] = self.parent.wT @ self._T
    #     else:
    #         self.__wT[:] = self._T

    #     self.__wq[:] = r2q(self.__wT[:3, :3], order="xyzs")

    def _propogate_scene(self):
        scene_graph_single(self.__scene)