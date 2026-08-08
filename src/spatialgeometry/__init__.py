from spatialgeometry.geom import (
    SceneNode,
    SceneGroup,
    Shape,
    Axes,
    Arrow,
    Path,
    CollisionShape,
    CollisionShapeGroup,
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
    "CollisionShapeGroup",
    "Mesh",
    "Cylinder",
    "Cuboid",
    "Box",
    "Sphere",
    "Axes",
    "Arrow",
    "Path",
    "SceneNode",
    "SceneGroup",
]

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("spatialgeometry")
except importlib.metadata.PackageNotFoundError:
    pass