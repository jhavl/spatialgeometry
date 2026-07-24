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

See the :doc:`api` page for the full class reference.
