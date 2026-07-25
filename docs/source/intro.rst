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


Quick start
===========

Create some shapes and give them a pose:

.. runblock:: pycon

    >>> import spatialgeometry as gm
    >>> from spatialmath import SE3
    >>> cube = gm.Cuboid([1, 1, 1], pose=SE3(0, 0, 0))
    >>> sphere = gm.Sphere(0.5, pose=SE3(2, 0, 0), color="red")
    >>> cube
    >>> sphere

Measure the distance between two shapes, and check whether they collide:

.. runblock:: pycon

    >>> import spatialgeometry as gm
    >>> from spatialmath import SE3
    >>> cube = gm.Cuboid([1, 1, 1], pose=SE3(0, 0, 0))
    >>> sphere = gm.Sphere(0.5, pose=SE3(2, 0, 0))
    >>> d, p1, p2 = cube.closest_point(sphere, inf_dist=10)
    >>> d
    >>> p1
    >>> p2
    >>> cube.iscollided(sphere)

A shape's pose and geometry can be serialised to a plain dict, used by
Swift to describe a scene to the browser:

.. runblock:: pycon

    >>> import spatialgeometry as gm
    >>> cube = gm.Cuboid([1, 1, 1])
    >>> cube.to_dict()


Displaying shapes
==================

Spatial Geometry itself has no renderer -- it describes geometry, it
doesn't draw it. To actually see a shape, add it to a `Swift
<https://github.com/jhavl/swift>`_ environment, which opens a browser tab
and renders whatever is added to it (robots and bare shapes alike):

.. code-block:: python

    # pip install swift-sim
    import spatialgeometry as gm
    from spatialmath import SE3
    from swift import Swift

    env = Swift()
    env.launch(realtime=True)

    cube = gm.Cuboid([1, 1, 1], pose=SE3(0, 0, 0.5), color="blue")
    sphere = gm.Sphere(0.3, pose=SE3(2, 0, 0.3), color="red")

    env.add(cube)
    env.add(sphere)

    env.hold()  # keep the browser tab open

Swift's ``env.add()`` accepts a bare ``Shape`` directly -- internally it
just calls the shape's ``to_dict()`` (shown above) and sends it over
a websocket to the browser. The same ``env.add()``/``env.step()`` pattern
works for a whole robot too; see Swift's own README for a worked example
moving a Panda arm towards a goal pose.

This example isn't executed when these docs are built (it needs a
browser and a running websocket connection), so treat it as a starting
point rather than verified-working output like the examples above.

See the :doc:`api` page for the full class reference.
