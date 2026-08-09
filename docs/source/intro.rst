************
Introduction
************

.. TODO: expand this page with more detail.

Spatial Geometry provides simple 3D shape primitives -- cuboids, cylinders,
spheres, and triangle meshes -- for representing robot links, obstacles, and
other geometry in a scene. Every shape carries a pose (position and
orientation) and an optional colour, and can be tested for distance and
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

Distance and collision checking need the ``collision`` extra (`Coal
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

    >>> import spatialgeometry as gm
    >>> from spatialmath import SE3
    >>> cube = gm.Cuboid([1, 2, 3], color="blue")
    >>> cube

In this case the cuboid is colored blue, and is 1 unit wide in the x-direction, 2 units deep in the
y-direction, and 3 units tall in the z-direction, and is centered at the origin. The
default pose is the identity transform, which places the shape at the origin with no
rotation. 

Spatial Geometry includes a number of primitive shapes such as cuboids, cylinders, spheres, as well
as triangle meshes. The following example creates a cuboid, a sphere, and a robot gripper from a mesh:

.. runblock:: pycon

    >>> import spatialgeometry as gm
    >>> from spatialmath import SE3
    >>> cube = gm.Cuboid([1, 2, 3], color="blue", pose=SE3(0, 0, 0))
    >>> sphere = gm.Sphere(0.5, pose=SE3(2, 0, 0.3), color="red")
    >>> gripper = gm.Mesh("../figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg")*SE3.Tx(0.5))
    >>> cube
    >>> sphere
    >>> gripper

Spatial Geometry uses the `trimesh <https://trimesh.org>`__ library to load triangle
meshes from a number of file formats, including glTF/GLB, PLY, STL,
OBJ, and Collada (``.dae``).

We can measure the distance between any two shapes, and check whether they collide:

.. runblock:: pycon
    :exclude: 1-5 

    >>> import spatialgeometry as gm
    >>> from spatialmath import SE3
    >>> cube = gm.Cuboid([1, 2, 3], color="blue", pose=SE3(0, 0, 0))
    >>> sphere = gm.Sphere(0.5, pose=SE3(2, 0, 0), color="red")
    >>> gripper = gm.Mesh("figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg"))
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

A shape's pose and geometry can be serialised to a plain dict, used by
Swift to describe a scene to the browser:

.. runblock:: pycon

    >>> import spatialgeometry as gm
    >>> cube = gm.Cuboid([1, 2, 3],color="blue", pose=SE3.Rx(90, unit="deg"))
    >>> cube.to_dict()


This shows the full list of properties that describe the shape, including its type, size, pose (as a translation
vector and unit quaternion), opacity, and color.

Visualizing shapes
==================

Spatial Geometry itself has no renderer -- it describes geometry but
doesn't draw it. To actually see a shape, we use the companion package `Swift
<https://github.com/jhavl/swift>`_.

Displaying shapes
-----------------

Swift opens a browser tab and renders whatever is added to the scene:

.. code-block:: python

    # pip install swift-sim
    import spatialgeometry as gm
    from spatialmath import SE3
    from swift import Swift

    env = Swift()
    env.launch(realtime=True)

    cube = gm.Cuboid([1, 2, 3], pose=SE3(0, 0, 0.5), color="blue")
    sphere = gm.Sphere(0.3, pose=SE3(2, 0, 0.3), color="red")
    gripper = gm.Mesh("../figs/panda_hand.dae", pose=SE3.Rx(90, unit="deg")*SE3.Tx(0.3))

    env.add(cube)
    env.add(sphere)
    env.add(gripper)

Swift's ``env.add()`` accepts a bare ``Shape`` directly -- internally it
just calls the shape's ``to_dict()`` (shown above) and sends it over
a websocket to the browser which runs Swift's JavaScript code to render the scene.

The scene is navigated with the mouse, using three.js's standard
`OrbitControls <https://threejs.org/docs/#examples/en/controls/OrbitControls>`__:

.. list-table:: Mouse controls
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Action
   * - Left button, drag
     - Rotate (orbit) the camera around the orbit target
   * - Right button, drag
     - Pan the camera and orbit target together
   * - Scroll wheel
     - Zoom in/out (dolly the camera towards/away from the orbit target)

The camera always looks at a fixed point in space called the *orbit target*
-- dragging with the left button rotates the camera around this point rather
than around the scene's origin. Swift sets the orbit target just above the
ground plane, at ``(0, 0, 0.2)``, so that rotating the view keeps your
shapes centred rather than swinging around the ground plane at ``z=0``.
Panning (right button or Ctrl/Cmd/Shift+left button) moves the orbit target itself, so subsequent
rotations pivot around wherever you've panned to.

Notes:

* The shapes have a finite z-displacement to lift them above the ground plane at
  z=0. The parts of objects below the ground plane are not visible from above the ground
  plane (default camera position) but if you rotate the scene using the mouse you can
  look beneath the ground and see the hidden part of the object.

 * While we can use a wide variety of mesh formats for Spatial Geometry, Swift only
   supports a subset of them: Collada (``.dae``) and STL (``.stl``). Collada (``.dae``)
   supports color and texture, which STL (``.stl``) does not. See Swift's README for
   more details.


See ``examples/displaying_shapes.py`` for a complete example.

Animating shapes
----------------

To animate a shape, we simply change its pose and call ``env.step()`` to update the
scene. The following example animates a sphere moving back and forth along the x-axis:

.. code-block:: python

    # pip install swift-sim
    import spatialgeometry as gm
    from spatialmath import SE3
    from swift import Swift

    env = Swift()
    env.launch(realtime=True)

    sphere = gm.Sphere(0.3, pose=SE3(0, 0, 0.3), color="red")

    env.add(sphere)

    for i in range(500):
        x = math.sin(i/20) * 0.5
        sphere.T = SE3.Trans(x, 0, 0.3)
        env.step(0.05)  # wait 0.05 seconds before next step

See ``examples/animating_shapes.py`` for a complete example.

Scene graphs
============

Spatial Geometry supports the concept of a scene graph, where shapes can be placed
relative to other shapes. Specifically, the child shape's pose is relative to the parent
shape's pose. This allows for hierarchical modeling of complex objects. 

Consider a household scene. We can model a table and specify its pose with respect to
the room. On the table we have several plates, placed relative to the table, and we have items
of food on each of the plates, placed relative to the plate. We can express this as
a directed acyclic graph (DAG), where each node represents an object in the scene.
In this context we call the graph a *scene graph*.

.. mermaid::

   graph LR
       Room[Room] --> Table[Table]
       Room --> Chair1[Chair1]
       Room --> Chair2[Chair2]
       Table --> Plate1((Plate1))
       Plate1 --> Beef[[Beef]]
       Table --> Plate2((Plate2))
       Plate2 --> Chicken[[Chicken]]

Each edge, an arrow from parent to child, represents a relative pose -- the pose of the
child relative to the parent. The pose of the table is specified relative to
the room, the pose of the plates is specified relative to the table, and so on.
We say that the table is a child of the room, and the room is the parent of the table.

The relative poses do not have to be constant -- they can be animated over time. For
example, we can animate the table moving around the room, and the plates and food will
move with it. We can also change the parent-child relationships over time, for example
if we pick up a plate and move it to a different table, or if we pick up a piece of food
and move it to a different plate. The scene graph is a powerful way to model complex
scenes with many objects and relationships.

We can also model an articulated robot arm in this way. Each link has a joint controlled
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

The following simple example creates a cube and attaches two spheres to it, at
specified offsets, and moves the cube within the scene:

.. runblock:: pycon

    import spatialgeometry as gm
    from spatialmath import SE3

    # Every Shape *is* a SceneNode
    cube = gm.Cuboid([1, 1, 1], color="blue", pose=SE3(0, 0, 0))
    sphere1 = gm.Sphere(0.5, pose=SE3(1, 0, 0), color="red")
    sphere2 = gm.Sphere(0.5, pose=SE3(0, 1, 0), color="green")
    # attach sphere1 and sphere2 to the cube, the relative offsets are given by their respective poses
    sphere1.scene_parent = cube
    sphere2.scene_parent = cube
    print(sphere1._wT[:3, 3])
    print(sphere2._wT[:3, 3])
    # Move the parent. This updates cube's own world pose but does NOT cascade to its children 
    cube.T = SE3(5, 0, 0)
    # Tell the scene graph that something has changed and that the world poses of all children need to be updated. 
    cube._propagate_scene_tree()
    # Now the children have been updated to reflect the new world pose of their parent
    print(sphere1._wT[:3, 3])
    print(sphere2._wT[:3, 3])

The ``_propagate_scene_tree()`` method is invoked automatically by Swift's ``env.step()`` method
to handle changes in object's pose. Because we are not using Swift in this example we need to call this manually.
This code structure can be scaled up indefintely to create complex scenes with many objects and relationships.
See ``examples/scene_graph.py`` for a similar example working in Swift.
    

See the :doc:`api` page for the full class reference.
