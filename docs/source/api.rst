*************
API Reference
*************

.. currentmodule:: spatialgeometry

Class summary
=============

Spatial geometry classes for 3D shapes and scene graph management.

.. autosummary::
   :toctree: generated

   SceneNode
   SceneGroup
   Shape
   CollisionShape
   CollisionShapeGroup

The class hierarchy for all Spatial Geometry classes is shown below:

.. inheritance-diagram::
   spatialgeometry.Shape
   spatialgeometry.Axes
   spatialgeometry.Arrow
   spatialgeometry.Path
   spatialgeometry.CollisionShape
   spatialgeometry.CollisionShapeGroup
   spatialgeometry.Mesh
   spatialgeometry.Cylinder
   spatialgeometry.Cuboid
   spatialgeometry.Sphere
   spatialgeometry.Ellipsoid
   spatialgeometry.Box
   spatialgeometry.SceneNode
   spatialgeometry.SceneGroup
   :parts: 1
   :top-classes: spatialgeometry.SceneNode, collections.UserList



Collision shapes
================

These are the basic 3D geometric shapes that can be rendered into a scene, and can also
be used for collision detection.

.. autosummary::

   Cuboid
   Sphere
   Ellipsoid
   Cylinder
   Mesh
   Box

These shapes all inherit from:

* the :class:`CollisionShape` base class which means they can be used for collision detection, and
* the :class:`SceneNode` base class which means they can be nodes in a scene graph to allow visualization and
  animation of complex scenes.

Collision shapes also support the collision operator ``&`` which returns True if the two shapes are colliding, and False otherwise. For example:

.. runblock:: pycon

   from spatialgeometry import Cuboid, Sphere
   from spatialmath import SE3

   c = Cuboid(scale=[1, 2, 3])
   s1 = Sphere(1, pose=SE3(4, 0, 0))
   s2 = Sphere(1, pose=SE3(0, 0, 0))

   c & s1
   c & s2 


.. autoclass:: Cuboid
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: collided

.. autoclass:: Sphere
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: collided

.. autoclass:: Cylinder
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: collided

.. autoclass:: Ellipsoid
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: collided

.. autoclass:: Box
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: collided

.. autoclass:: Mesh
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: collided



Shapes
======

These are the basic 3D geometric shapes that can be rendered into a scene, but they cannot
be used for collision detection. 

.. autosummary::

   Axes
   Arrow
   Path
   
They all inherit directly from the :class:`Shape` base class.

.. autoclass:: Axes
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: Arrow
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: Path
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


Scene Graphs
============

.. autoclass:: SceneNode
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


.. autoclass:: SceneGroup
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: CollisionShapeGroup
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: collided