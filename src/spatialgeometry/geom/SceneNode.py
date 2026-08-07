#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from __future__ import annotations

from numpy import ndarray, eye, copy as npcopy, array
from spatialmath.base import r2q
from spatialgeometry.scene import node_init, node_update, scene_graph_children, scene_graph_tree
from spatialmath import SE3
from copy import deepcopy


class SceneNode:
    def __init__(
        self,
        T: ndarray = eye(4),
        scene_parent: SceneNode | None = None,
        scene_children: list[SceneNode] | None = None,
    ) -> None:
        # These three are static attributes which can never be changed
        # If these are directly accessed and re-written, segmentation faults
        # will follow very soon after
        # wT and sT cannot be accessed and set by users by base can be
        # modified through its setter

        # The world transform
        self.__wT = eye(4).copy(order="F")

        # The quaternion extracted from wT
        self.__wq = array([0.0, 0.0, 0.0, 1.0])

        # The local transform
        self.__T = eye(4).copy(order="F")
        self.__T[:] = T.copy(order="F")

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

        # Update scene tree
        self._propogate_scene_children()

    # --------------------------------------------------------------------- #

    def _custom_scene_node_init(
        self,
        T: ndarray = eye(4),
        scene_parent: SceneNode | None = None,
        scene_children: list[SceneNode] | None = None,
    ) -> None:
        # The world transform
        self.__wT = eye(4).copy(order="F")

        # The quaternion extracted from wT
        self.__wq = array([0.0, 0.0, 0.0, 1.0])

        # The local transform
        self.__T = eye(4).copy(order="F")
        self.__T[:] = T.copy(order="F")

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

        # Update scene tree
        self._propogate_scene_children()

    # --------------------------------------------------------------------- #

    def __init_c(self):
        """
        Super Private method which initialises a C object to hold Data

        """

        return node_init(
            len(self._scene_children),
            self.__T,
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

    def __copy__(self):
        return deepcopy(self)

    def __deepcopy__(self, memo):
        result = SceneNode(
            T=self._T,
        )

        result._scene_children = self.scene_children.copy()
        result._scene_parent = self.scene_parent
        result.__update_c()
        memo[id(self)] = result
        return result

    def __str__(self) -> str:
        return f"{type(self).__name__} at {SE3(self._T, check=False).strline()}"

    # --------------------------------------------------------------------- #

    @property
    def scene_parent(self) -> SceneNode | None:
        """
        Returns the parent node of this object

        """
        return self._scene_parent

    @scene_parent.setter
    def scene_parent(self, parent: SceneNode) -> None:
        """
        Sets a new parent node of this object, will automatically update
        the parents child

        """
        # Set our parent (also validates this won't create a cycle)
        self._update_scene_parent(parent)

        # Update our parents children
        parent._update_scene_children(self)

    def _update_scene_parent(self, parent: SceneNode) -> None:
        """
        Sets a new parent node of this object, does NOT update
        the parents child

        """
        # A node can't become its own ancestor -- walk the new parent's own
        # chain of parents; if self shows up (or parent is self), this
        # reparenting would create a cycle. Nothing downstream checks for
        # this: _propogate_scene_tree()'s root-finding walk (SceneNode.py,
        # also mirrored in scene.py and the compiled scene_nb.cpp) has no
        # cycle detection of its own -- a parent-chain cycle makes it spin
        # forever (an infinite loop, not a catchable exception), and
        # Swift's env.step() calls it every step. O(depth) per call here;
        # fine for how small these graphs actually get, see tech-debt
        # issue for a cheaper approach if that ever stops being true.
        ancestor = parent
        while ancestor is not None:
            if ancestor is self:
                raise ValueError(
                    f"Cannot set {self!r}'s scene_parent to {parent!r} -- "
                    f"{parent!r} is already a descendant of {self!r}, this "
                    "would create a cycle in the scene graph"
                )
            ancestor = ancestor.scene_parent

        self._scene_parent = parent

        # Update c
        self.__update_c()

    # --------------------------------------------------------------------- #

    @property
    def scene_children(self) -> list[SceneNode]:
        """
        Returns the child nodes of this object

        """
        return self._scene_children

    @scene_children.setter
    def scene_children(self, children: list[SceneNode]) -> None:
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

    def _update_scene_children(self, child: SceneNode) -> None:
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

    # --------------------------------------------------------------------- #

    @property
    def _T_reference(self) -> ndarray:
        """
        Returns the transform of this object with respect to the parent
        frame.

        """
        return self.__T

    @property
    def _T(self) -> ndarray:
        """
        Returns a copy of the transform of this object with respect to the parent
        frame.

        """
        return npcopy(self.__T)

    @_T.setter
    def _T(self, T: ndarray):
        self.__T[:] = T.copy(order="F")

        if self._scene_parent is not None:
            self.__wT[:] = self._scene_parent._wT @ self._T
        else:
            self.__wT[:] = self._T

        self.__wq[:] = r2q(self.__wT[:3, :3], order="xyzs")

    @property
    def T(self) -> ndarray:
        return self._T

    @T.setter
    def T(self, T_new: ndarray | SE3) -> None:
        if isinstance(T_new, SE3):
            T_new = T_new.A
        self._T = T_new

    # --------------------------------------------------------------------- #
    # Scene transform propogation methods
    #
    # The scene graph is a Forest -- A disjoint union of Rooted Trees
    # Each tree has a single root, no cycles, and each node has at most one
    # parent but unlimited children.
    # --------------------------------------------------------------------- #

    def _propogate_scene_children(self):
        """
        Propogates the world transform starting from this node going downwards
        through the tree (will not go through parents)
        """
        scene_graph_children(self.__scene)

    def _propogate_scene_tree(self):
        """
        Propogates the world transform starting from this root of the tree in
        which this node lives
        """
        scene_graph_tree(self.__scene)

    # --------------------------------------------------------------------- #

    def attach(self, object: SceneNode) -> None:
        new_childs = self.scene_children
        new_childs.append(object)
        self.scene_children = new_childs

    def attach_to(self, object: SceneNode) -> None:
        self.scene_parent = object
