from spatialgeometry import Cuboid, Cylinder, Sphere
from spatialmath import SE3

# define the room, table, plates and food items
room = Cuboid([5, 5, 3], color="gray", pose=SE3(0, 0, 1.5))
table = Cuboid([2, 1, 1], pose=SE3(1, 0, 0.5)) # 2x1m table that is 1m tall
plate1 = Cylinder(0.2, 0.04, color="white", pose=SE3(0.5, 0, 0.02))
plate2 = Cylinder(0.2, 0.04, color="white", pose=SE3(-0.5, 0, 0.02))
beef = Sphere(0.05, color="saddlebrown",  pose=SE3(0.05, 0, 0.05))
chicken = Sphere(0.05, color="peru",      pose=SE3(0, 0.04, 0.05))

# set up the scene graph by setting the scene_parent of each shape
table.scene_parent = room
plate1.scene_parent = table
plate2.scene_parent = table
beef.scene_parent = plate1
chicken.scene_parent = plate2

print("Scene graph:")
print(plate1.tree())

print("\nPose of chicken (initial):", SE3(chicken._wT).strline())
room.update() 
print("Pose of chicken (updated):", SE3(chicken._wT).strline())

table.T = table.T * SE3(0.2, 0.3, 0) # move the table
room.update() # update the scene graph starting from the root node
print("Pose of chicken (moved table):", SE3(chicken._wT).strline())

plate2.T = plate2.T * SE3.Rz(45, "deg") # rotate the plate
room.update() # update the scene graph starting from the root node
print("Pose of chicken (rotated plate):", SE3(chicken._wT).strline())


