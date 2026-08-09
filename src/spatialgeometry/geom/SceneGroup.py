#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from __future__ import annotations

from collections import UserList
from collections.abc import Iterable

from spatialgeometry.geom.SceneNode import SceneNode


class SceneGroup(SceneNode, UserList):
    """
    An ordered, list-like collection of :class:`SceneNode` objects (nodes can
    be nested groups, shapes, or any other :class:`SceneNode` subclass) that
    itself behaves like a single :class:`SceneNode`.

    :param initlist: Initial elements to populate the group with.

    A :class:`SceneGroup` is itself a :class:`SceneNode` so its elements can be
    collectively, parented or nested like any other node in the scene graph. This class
    inherits from :class:`collections.UserList` so it behaves like a Python list, but it
    is not a subclass of :class:`list` so it can be subclassed itself.

    Every element's ``scene_parent`` is kept pointing at the group -- adding an
    element (via the constructor, ``append``, ``extend``, ``insert``, or item
    assignment) sets it, and removing one (``remove``, ``pop``, ``clear``,
    ``del``) clears it back to ``None``. This is what makes moving the group
    move its elements with it.

    The reverse holds too, and isn't just a side effect -- ``append()`` is
    nothing more than ``item.scene_parent = self``, so setting any node's
    ``scene_parent`` to a :class:`SceneGroup` directly (with no ``append``
    call at all) equally makes it a member of that group's list. List
    membership and scene-graph parentage are the same relationship, not
    two things kept in sync -- there's no way for them to disagree.

    .. runblock:: pycon

        >>> from spatialgeometry import SceneGroup, Cuboid, Sphere
        >>> from spatialmath import SE3
        >>> group = SceneGroup()
        >>> print(len(group))
        >>> group.append(Cuboid([1,2,3]))
        >>> group.append(Sphere(1, pose=SE3.Trans(1,0,0)))
        >>> print(len(group))
    """

    def __init__(
        self, initlist: Iterable[SceneNode] | None = None, **kwargs
    ) -> None:
        super().__init__(
            scene_children=list(initlist) if initlist is not None else None,
            **kwargs,
        )

    def __getitem__(self, i: int) -> SceneNode:
        return self._scene_children[i]

    def __setitem__(self, i: int, item: SceneNode) -> None:
        if isinstance(i, slice):
            raise NotImplementedError(
                "Slice assignment is not supported for SceneGroup"
            )
        old = self._scene_children[i]
        self.remove(old)
        self.insert(i, item)

    def __delitem__(self, i: int | slice) -> None:
        if isinstance(i, slice):
            for item in list(self._scene_children[i]):
                self.remove(item)
        else:
            self.remove(self._scene_children[i])

    def append(self, item: SceneNode) -> None:
        """Add ``item`` to the end of the group and set its ``scene_parent``
        to this group."""
        item.scene_parent = self

    def extend(self, other: Iterable[SceneNode]) -> None:
        """Add every element of ``other`` to the end of the group, setting
        each one's ``scene_parent`` to this group."""
        for item in other:
            self.append(item)

    def insert(self, i: int, item: SceneNode) -> None:
        """Insert ``item`` at position ``i`` and set its ``scene_parent`` to
        this group."""
        item.scene_parent = self
        self._scene_children.insert(i, self._scene_children.pop())

    def remove(self, item: SceneNode) -> None:
        """Remove ``item`` from the group and clear its ``scene_parent``."""
        self._remove_scene_child(item)
        item._scene_parent = None

    def pop(self, i: int = -1) -> SceneNode:
        """Remove and return the element at position ``i`` (default: the
        last one), clearing its ``scene_parent``."""
        item = self._scene_children[i]
        self.remove(item)
        return item

    def clear(self) -> None:
        """Remove every element from the group, clearing each one's
        ``scene_parent``."""
        for item in list(self._scene_children):
            self.remove(item)

    @property
    def data(self) -> list[SceneNode]:
        return self._scene_children

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.data!r})"
