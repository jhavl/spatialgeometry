from spatialgeometry import Cuboid, Mesh
from spatialmath import SE3
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=80)

gripper = Mesh("../docs/figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg")*SE3.Tx(0.5))

print(f"dimensions of bounding box:\n{gripper.extents()}")
print(f"bounds of bounding box:\n{gripper.bounds()}")
print(f"corners of bounding box:\n{gripper.corners()}")
