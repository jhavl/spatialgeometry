************
Introduction
************

Spatial Geometry provides simple 3D shape primitives --- cuboids, cylinders,
spheres, ellipsoids, triangular meshes, axes and paths --- for representing robot links, obstacles, and
other geometry in a scene. Every shape carries a pose (position and
orientation) and optional rendering properties such as color and opacity.  A shape can be tested for distance and
collision against any other shape using `Coal
<https://github.com/coal-library/coal>`_.

It's used by the `Robotics Toolbox for Python
<https://github.com/petercorke/robotics-toolbox-python>`_ to describe robot
link geometry, and by `Swift <https://github.com/jhavl/swift>`_ to render
scenes in the browser.


Installation
============

::

    pip install spatialgeometry

Distance and collision checking requires the ``collision`` extra (`Coal
<https://github.com/coal-library/coal>`_ and `trimesh <https://trimesh.org>`_)::

    pip install spatialgeometry[collision]

If you don't need collision checking but do want to work with mesh files --
e.g. :meth:`~spatialgeometry.Mesh.corners`/``bounds``/``extents`` for a
mesh's bounding box, which reads the file via trimesh but never touches Coal
at all -- install just that piece::

    pip install spatialgeometry[mesh]

Unlike ``collision``, this works on Windows too: trimesh has no Windows-wheel
problem of its own, only Coal does.


Quick start
===========


Create a cuboid (rectangular prism) shape:

.. runblock:: pycon

    >>> from spatialgeometry import Cuboid
    >>> from spatialmath import SE3
    >>> cube = Cuboid([1, 2, 3], color="blue")
    >>> cube

In this case the cuboid is colored blue, and is 1 unit wide in the x-direction, 2 units deep in the
y-direction, and 3 units tall in the z-direction, and is centered at the origin. The
default pose is the identity transform, which places the shape at the origin with no
rotation. 

Spatial Geometry includes a number of primitive shapes such as cuboids, cylinders, spheres,  ellipsoids, triangular meshes, axes and paths.
The following example creates a cuboid, a sphere, and a robot gripper from a mesh:

.. runblock:: pycon

    >>> from spatialgeometry import Cuboid, Sphere, Mesh
    >>> from spatialmath import SE3
    >>> cube = Cuboid([1, 2, 3], color="blue", pose=SE3(0, 0, 0))
    >>> sphere = Sphere(0.5, pose=SE3(2, 0, 0.3), color="red")
    >>> gripper = Mesh("figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg")*SE3.Tx(0.5))
    >>> cube
    >>> sphere
    >>> gripper

Spatial Geometry uses the `trimesh <https://trimesh.org>`__ library to load triangular
meshes from a number of file formats --- see :doc:`fileformats` for the full list, and
which of them also work for display in `Swift <https://github.com/jhavl/swift>`_.

We can determine the axis-aligned bounding boxes of any shape, for example:

.. runblock:: pycon
    :exclude: 1-5

    >>> from spatialgeometry import Cuboid, Sphere, Mesh
    >>> from spatialmath import SE3
    >>> cube = Cuboid([1, 2, 3], color="blue", pose=SE3(0, 0, 0))
    >>> sphere = Sphere(0.5, pose=SE3(2, 0, 0.3), color="red")
    >>> gripper = Mesh("figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg")*SE3.Tx(0.5))
    >>> cube.extents() # dimensions of the bounding box
    >>> cube.bounds() # min and max coordinates of the bounding box
    >>> sphere.extents() # dimensions of the bounding box
    >>> sphere.bounds() # min and max coordinates of the bounding box
    >>> gripper.extents() # dimensions of the bounding box
    >>> gripper.bounds() # min and max coordinates of the bounding box
    >>> gripper.corners() # coordinates of the 8 corners of the bounding box

The edges of the box are aligned with the x-, y- and z-axes of the world frame.
The ``extents`` method returns the dimensions of the bounding box, which clearly reflects the constructed dimensions of the cuboid and sphere.
The ``bounds`` method returns the minimum and maximum coordinates of the bounding box in the local frame.
The bounds show that the cuboid and sphere are centred about the origin -- this is true for all SpatialGeometry shape primitives
but not necessarily true for meshes.
For all methods, the rows correspond to the x-, y-, and z-axes.
This bounding box is computed in the object's local frame, before any transformation (``pose`` parameter at construction, or the ``T`` attribute set).

To determine the axis-aligned bounding box in the world frame we pass the ``world=True`` argument to the methods:

.. runblock:: pycon
    :exclude: 1-2

    >>> from spatialgeometry import Cuboid
    >>> from spatialmath import SE3
    >>> cube = Cuboid([1, 2, 3], color="blue", pose=SE3(10, 11, 12)*SE3.RPY(10, 20, 30, unit="deg"))
    >>> cube.extents(world=True) # dimensions of the bounding box in the world frame
    >>> cube.bounds(world=True) # min and max coordinates of the bounding box in the world frame
    >>> cube.corners(world=True) # coordinates of the 8 corners of the bounding box in the world frame


We clearly see that the bounding box in the world frame is larger than the bounding box in the
local frame, because the shape has been rotated in the world frame, and the corners reflect that the 
shape has been translated in the world frame.


We can measure the distance between any two shapes, and check whether they collide:

.. runblock:: pycon
    :exclude: 1-5 

    >>> from spatialgeometry import Cuboid, Sphere, Mesh
    >>> from spatialmath import SE3
    >>> cube = Cuboid([1, 2, 3], color="blue", pose=SE3(0, 0, 0))
    >>> sphere = Sphere(0.5, pose=SE3(2, 0, 0), color="red")
    >>> gripper = Mesh("figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg"))
    >>> d, p1, p2 = cube.closest_point(sphere, inf_dist=10)
    >>> d
    >>> p1 # point on cube
    >>> p2 # point on sphere
    >>> cube.iscollided(sphere)
    >>> d, p1, p2 = gripper.closest_point(sphere, inf_dist=10)
    >>> d
    >>> p1 # point on gripper   
    >>> p2 # point on sphere
    >>> gripper.iscollided(sphere)

``p1`` and ``p2`` are the closest points on each shape, and ``d`` is the distance
between them. If the shapes collide, ``d`` is zero and ``p1`` and ``p2`` are the same
point.




A shape can be completely described by a dict:

.. runblock:: pycon

    >>> from spatialgeometry import Cuboid
    >>> cube = Cuboid([1, 2, 3],color="blue", pose=SE3.Rx(90, unit="deg"))
    >>> cube.to_dict()


which includes the full list of properties that describe the shape, including its type, size, pose (as a translation
vector and unit quaternion), opacity, and color. This is used by
Swift to describe objects to the JavaScript code running in the browser.

Shape properties
================

The properties of an object are represented by read/write properties.
We can change a shape's geometric properties, such as size or pose, as well as
visual properties, such as color and opacity, by setting the appropriate property. For example:

.. runblock:: pycon

    >>> from spatialgeometry import Cuboid
    >>> from spatialmath import SE3
    >>> cube = Cuboid([1, 2, 3], color="blue")
    >>> cube
    >>> cube.color = "red"
    >>> cube.T = SE3(1, 2, 3)*SE3.Rx(90, unit="deg")
    >>> cube.scale = [2, 4, 6]
    >>> cube.opacity = 0.5
    >>> cube
    >>> print(cube)


Scene graphs
============

Spatial Geometry supports scene graphs --- shapes can be placed in the scene
relative to other shapes. If the *parent* shape's pose is changed, the pose of all the *child* shapes is updated accordingly.
This allows for hierarchical modeling of complex scenes. 

Consider a household scene. We model a room that contains a table on which are placed several plates, and each plate has an item of food.
We can express this as
a directed acyclic graph (DAG) --- *scene graph* --- where each node represents an object in the scene:

.. mermaid::

   graph LR
       Room[Room] --> Table[Table]
       Room --> Chair1[Chair1]
       Room --> Chair2[Chair2]
       Table --> Plate1((Plate1))
       Plate1 --> Beef[[Beef]]
       Table --> Plate2((Plate2))
       Plate2 --> Chicken[[Chicken]]

Each edge, an arrow from parent to child, represents a relative pose --- the pose of the
child relative to the parent. The pose of the table is specified relative to the room,
the pose of the plates is specified relative to the table, and so on. We say that the
table is a child of the room, and the room is the parent of the table. If we move the
table, the plates and food move with it. If we move a plate, the food on it moves with
it. The scene graph allows us to model the relationships between objects in a scene in a concise way.


.. runblock:: pycon

    from spatialgeometry import Cuboid, Cylinder, Sphere
    from spatialmath import SE3

    room = Cuboid([5, 5, 3], color="gray", pose=SE3(0, 0, 1.5))
    table = Cuboid([2, 1, 1], pose=SE3(1, 0, 0.5)) # 2x1m table that is 1m tall
    plate1 = Cylinder(0.2, 0.04, color="white", pose=SE3(0.5, 0, 0.02))
    plate2 = Cylinder(0.2, 0.04, color="white", pose=SE3(-0.5, 0, 0.02))
    beef = Sphere(0.05, color="saddlebrown",  pose=SE3(0.05, 0, 0.05))
    chicken = Sphere(0.05, color="peru",      pose=SE3(0, 0.04, 0.05))
    table.scene_parent = room   
    plate1.scene_parent = table
    plate2.scene_parent = table
    beef.scene_parent = plate1
    chicken.scene_parent = plate2

    print(plate1.tree())
    print(chicken._wT) # world pose of chicken

The ``tree()`` method prints the scene graph in a human-readable form, showing the parent-child relationships between the shapes.

.. note::
    The parent relationships can also be set at construction time by passing the ``scene_parent`` argument to the constructor of a shape.
    This does require that the parent shape is constructed first, so that it can be passed to the child shape's constructor.


The initial world pose of the chicken is printed, and we see that it is the same as its
local pose --- the scene graph has not yet been updated to reflect the relative
poses of its parents. 
The ``_wT`` property is a read-only property that returns the world pose of the shape as a 4x4 Numpy array.

We can update the scene graph by calling the ``update()`` method on the root node of the
scene graph (the room in this case):

.. runblock:: pycon
    :exclude: 1-14

    from spatialgeometry import Cuboid, Cylinder, Sphere
    from spatialmath import SE3
    room = Cuboid([5, 5, 3], color="gray", pose=SE3(0, 0, 1.5))
    table = Cuboid([2, 1, 1], pose=SE3(1, 0, 0.5)) # 2x1m table that is 1m tall
    plate1 = Cylinder(0.2, 0.04, color="white", pose=SE3(0.5, 0, 0.02))
    plate2 = Cylinder(0.2, 0.04, color="white", pose=SE3(-0.5, 0, 0.02))
    beef = Sphere(0.05, color="saddlebrown",  pose=SE3(0.05, 0, 0.05))
    chicken = Sphere(0.05, color="peru",      pose=SE3(0, 0.04, 0.05))
    table.scene_parent = room   
    plate1.scene_parent = table
    plate2.scene_parent = table
    beef.scene_parent = plate1
    chicken.scene_parent = plate2
    print(chicken._wT)
    room.update() # update the scene graph starting from the root node
    print(chicken._wT)

The beauty of scene graphs is that they allow us to model complex scenes with many
objects and relationships in a concise way. The relative poses of the child shapes are
specified relative to their parent shapes, and the world poses of all shapes can be
computed by traversing the scene graph from the root node down to the leaves. The
relative poses do not have to be constant --- they can be animated over time. For
example, we can animate the table moving around the room, and the plates and food will
move with it. We can also change the parent-child relationships over time, for example
if we pick up a plate and move it to a different table, or if we pick up a piece of food
and move it to a different plate. The scene graph is a powerful way to model complex
scenes with many objects and relationships.

Let's demonstrate this by moving the table:

.. runblock:: pycon
    :exclude: 1-14

    from spatialgeometry import Cuboid, Cylinder, Sphere
    from spatialmath import SE3
    room = Cuboid([5, 5, 3], color="gray", pose=SE3(0, 0, 1.5))
    table = Cuboid([2, 1, 1], pose=SE3(1, 0, 0.5)) # 2x1m table that is 1m tall
    plate1 = Cylinder(0.2, 0.04, color="white", pose=SE3(0.5, 0, 0.02))
    plate2 = Cylinder(0.2, 0.04, color="white", pose=SE3(-0.5, 0, 0.02))
    beef = Sphere(0.05, color="saddlebrown",  pose=SE3(0.05, 0, 0.05))
    chicken = Sphere(0.05, color="peru",      pose=SE3(0, 0.04, 0.05))
    table.scene_parent = room   
    plate1.scene_parent = table
    plate2.scene_parent = table
    beef.scene_parent = plate1
    chicken.scene_parent = plate2
    print(chicken._wT)
    table.T = table.T * SE3(0.2, 0.3, 0) # move the table
    room.update() # update the scene graph starting from the root node
    print(chicken._wT)

We can see that the world pose of the chicken has changed, reflecting the movement of the table.
See ``examples/room.py`` for a similar example working in Swift.

We can use this same approach to model an articulated robot arm. Each link has a joint controlled
pose relative to its parent link and we have created a scene graph link from the robot's
gripper to a piece of food that it is holding:
    
.. mermaid::

   graph LR
       Base[Base] --> Link1([Link1])
       Link1 --> Link2([Link2])
       Link2 --> Link3([Link3])
       Link3 --> Gripper{{Gripper}}
       Gripper{{Gripper}} --> Beef[[Beef]]

In code the parent-child relationships are expressed by setting the ``scene_parent``
property of a shape to another shape. Each ``Shape`` subclass can be a scene node in the
graph. 


See the :doc:`api` page for the full class reference.

Scene groups
------------

All shapes, collision and non-collision shapes, inherit from :class:`Shape` which
inherits from the :class:`SceneNode` base class. The :class:`SceneGroup` class is a
special type of :class:`SceneNode` which inherits
list-like properties from ``UserList``. It can be used to group several shapes together --- they
move as a single object when the :class:`SceneGroup` is moved, and they have a common parent.  Since 
a :class:`SceneGroup` inherits from :class:`SceneNode` it can have child nodes.

.. runblock:: pycon

    from spatialgeometry import Cuboid, Sphere, SceneGroup
    from spatialmath import SE3

    cube = Cuboid([1, 1, 1], color="blue", pose=SE3(0, 0, 0))
    sphere1 = Sphere(0.5, pose=SE3(1, 0, 0), color="red")
    sphere2 = Sphere(0.5, pose=SE3(0, 1, 0), color="green")
    sphere3 = Sphere(0.5, pose=SE3(0, 0, 1), color="blue")
    sphere4 = Sphere(0.5, pose=SE3(1, 1, 0), color="white")
    group = SceneGroup([sphere1, sphere2])
    group.scene_parent = cube
    sphere3.scene_parent = group
    sphere4.scene_parent = cube
    group
    print(cube.tree())

Note that ``sphere3`` was never passed to ``group.append()`` -- it was attached by
setting its ``scene_parent`` directly to ``group`` -- yet it still shows up as a
member of ``group``. This isn't a special case: ``append()`` *is* just
``item.scene_parent = self``, so the two are the same operation under different
names. List membership and scene-graph parentage of a :class:`SceneGroup` are the
same relationship, not two things kept in sync with each other.

A related concept is the :class:`CollisionShapeGroup` which groups together
:class:`CollisionShape` instances. In robotics, the *collision shape* of a robot link is
typically a collection of simple 3D primitives (cuboids, spheres, cylinders) that are
used for fast and efficient collision detection. The more detailed and accurate
triangular meshes are only used for visualization.  A collision check that involves one
or two collision shape groups will check for collisions between all the shapes in each
group.

Visualizing shapes
==================

Spatial Geometry itself has no renderer --- it describes geometry but
doesn't draw it. To actually see a shape, we use the companion package `Swift
<https://github.com/jhavl/swift>`_.  Here's a simple example that creates a cuboid, a sphere, and a robot gripper from a mesh, and displays them:


.. code-block:: python

    # pip install swift-sim
    from spatialgeometry import Cuboid, Sphere, Mesh
    from spatialmath import SE3
    from swift import Swift

    env = Swift()
    env.launch(realtime=True)

    cube = Cuboid([1, 2, 3], pose=SE3(0, 0, 0.5), color="blue")
    sphere = Sphere(0.3, pose=SE3(2, 0, 0.3), color="red")
    gripper = Mesh("../figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg")*SE3.Tx(0.3))

    env.add(cube)
    env.add(sphere)
    env.add(gripper)

Swift opens a new  browser tab and renders whatever is *added* to the scene.  If the
pose of a shape is changed the ``env.step()`` will update the appearance of the scene in the browser.


More information about Swift and its capabilities for animation can be found in the
`Swift documentation <https://jhavl.github.io/swift/>`_.
