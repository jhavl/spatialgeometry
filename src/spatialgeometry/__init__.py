from spatialgeometry.geom import (
    SceneNode,
    SceneGroup,
    Shape,
    Axes,
    Arrow,
    Path,
    CollisionShape,
    Mesh,
    Cylinder,
    Cuboid,
    Box,
    Sphere,
    Ellipsoid)

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
    "Ellipsoid",
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