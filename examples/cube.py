from spatialgeometry import Cuboid
from spatialmath import SE3
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=80)

cube = Cuboid([1, 2, 3], color="blue")
print(cube)
print("pose of cube:", SE3(cube.T).strline())
print("\nIn local frame:")
print(f"dimensions of bounding box:\n{cube.extents()}")
print(f"bounds of bounding box:\n{cube.bounds()}")
print(f"corners of bounding box:\n{cube.corners()}")
print()
cube.T = SE3(10, 11, 12)*SE3.RPY(10, 20, 30, unit="deg")
print("pose of cube:", SE3(cube.T).strline())

print("\nIn world frame:")
print(f"dimensions of bounding box:\n{cube.extents(world=True)}")
print(f"bounds of bounding box:\n{cube.bounds(world=True)}")
print(f"corners of bounding box:\n{cube.corners(world=True)}")

