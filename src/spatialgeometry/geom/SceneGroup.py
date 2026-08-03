#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from __future__ import annotations

from collections import UserList

from spatialgeometry.geom.SceneNode import SceneNode


class SceneGroup(SceneNode, UserList):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __getitem__(self, i: int) -> SceneNode:
        return self._scene_children[i]

    @property
    def data(self) -> list[SceneNode]:
        return self._scene_children

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.data!r})"
