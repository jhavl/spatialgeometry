from spatialgeometry.geom import (
    SceneNode,
    SceneGroup,
    Shape,
    Axes,
    Arrow,
    CollisionShape,
    Mesh,
    Cylinder,
    Cuboid,
    Box,
    Sphere)

from spatialgeometry import tools


__all__ = [
    # aliased
    "tools",
    # geom
    "Shape",
    "CollisionShape",
    "Mesh",
    "Cylinder",
    "Cuboid",
    "Box",
    "Sphere",
    "Axes",
    "Arrow",
    "SceneNode",
    "SceneGroup",
]

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("spatialgeometry")
except importlib.metadata.PackageNotFoundError:
    pass