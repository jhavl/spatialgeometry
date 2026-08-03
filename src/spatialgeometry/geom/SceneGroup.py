#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from __future__ import annotations

from collections import UserList

from spatialgeometry.geom.SceneNode import SceneNode


class SceneGroup(SceneNode, UserList):
    """
    An ordered, list-like collection of :class:`SceneNode` objects.

    A :class:`SceneGroup` is itself a :class:`SceneNode` so its elements can be
    collectively, parented or nested like any other node in the scene graph. This class
    inherits from :class:`collections.UserList` so it behaves like a Python list, but it
    is not a subclass of :class:`list` so it can be subclassed itself.

    .. runblock:: pycon

        >>> from spatialgeometry import SceneGroup, Cuboid, Sphere
        >>> from spatialmath import SE3
        >>> group = SceneGroup()
        >>> print(len(group))
        >>> group.append(Cuboid([1,2,3]))
        >>> group.append(Sphere(1, pose=SE3.Trans(1,0,0)))
        >>> print(len(group))
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __getitem__(self, i: int) -> SceneNode:
        return self._scene_children[i]

    @property
    def data(self) -> list[SceneNode]:
        return self._scene_children

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.data!r})"
