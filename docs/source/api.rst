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

The class hierarchy for all Spatial Geometry classes is shown below:

.. inheritance-diagram::
   spatialgeometry.Shape
   spatialgeometry.Axes
   spatialgeometry.Arrow
   spatialgeometry.Path
   spatialgeometry.CollisionShape
   spatialgeometry.Mesh
   spatialgeometry.Cylinder
   spatialgeometry.Cuboid
   spatialgeometry.Sphere
   spatialgeometry.Box
   spatialgeometry.SceneNode
   spatialgeometry.SceneGroup
   :parts: 1
   :top-classes: spatialgeometry.SceneNode, collections.UserList



Collision shapes
================

These are the basic 3D geometric shapes that can be embedded in a scene, and can also
be used for collision detection.

.. autosummary::

   Mesh
   Cylinder
   Cuboid
   Sphere
   Box

These shapes all inherit from:

* the :class:`CollisionShape` base class which means they can be used for collision detection, and
* the :class:`SceneNode` base class which means they can be nodes in a scene graph to allow visualization and
  animation of complex scenes.


.. autoclass:: Mesh
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: Cylinder
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: Cuboid
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: Sphere
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: Box
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:



Shapes
======

These are the basic 3D geometric shapes that can be embedded in a scene, but they cannot
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



Scene graphs
============

A scene graph is a hierarchical structure of nodes that represent objects in a 3D scene.
Each node can have child nodes, allowing for complex scenes to be built from simpler
components.  The individual nodes can be moved, rotated, and scaled, and these transformations are inherited by their child nodes.

.. autosummary::

   SceneNode
   SceneGroup
