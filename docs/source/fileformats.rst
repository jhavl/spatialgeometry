*****************
Mesh File Formats
*****************

A :class:`~spatialgeometry.Mesh` shape's ``filename`` is read by two
independent consumers, and each only understands a limited set of all possible mesh file
formats:

* **The browser**, via `Swift <https://github.com/jhavl/swift>`_. Swift is
  built on `three.js <https://threejs.org>`__, and picks a loader based on
  the file's extension to display the mesh.
* **trimesh**, used two separate ways:

  - Feeding `Coal <https://github.com/coal-library/coal>`_ for distance/collision
    queries -- lazily, the first time a collision-enabled shape (``collision=True``)
    is actually used in a ``closest_point()`` or ``iscollided()`` call. Coal itself
    doesn't read files at all, it just consumes the vertices and triangles that
    trimesh hands it.
  - Computing a mesh's local bounding box (``corners()``/``bounds()``/``extents()``)
    directly from trimesh's own ``mesh.bounds`` -- independent of ``collision``,
    and never cached, so each call reloads the file.

If you want a mesh to both **display in Swift** and **participate in
collision checking**, the file needs to be loadable by both. If you only
need one or the other, the constraint relaxes: use anything three.js
supports for display-only meshes (``collision=False``), or anything
trimesh supports for collision-only meshes never rendered on screen.

`Paul Bourke's Data Formats page <https://paulbourke.net/dataformats/>`_ is a great
reference for the history and capabilities of many of the formats discussed here.


The two sets, and their intersection
=====================================

Swift's loaders cover ``.dae``, ``.stl``, ``.obj`` (+ matching ``.mtl``),
``.gltf``/``.glb``, ``.ply``, ``.wrl`` (VRML), and ``.pcd`` (point clouds).

`trimesh <https://trimesh.org>`__ (pulled in by SpatialGeometry's ``collision`` extra) covers ``.dae``, ``.stl``,
``.obj``, ``.gltf``/``.glb``, ``.ply``, ``.off``, ``.xyz``, ``.zae``, and a
few CAD-interchange formats (``.3mf``, ``.step``/``.stp``) if the relevant
optional dependencies are present.

trimesh itself has further `install extras <https://trimesh.org/install.html>`__
(``pip install trimesh[easy]``, ``trimesh[recommend]``, ...) that unlock those
"if the relevant optional dependencies are present" formats -- but installing
them doesn't widen the overlap below, because none of the formats they add are
formats Swift's three.js loaders understand in the first place:

* ``trimesh[easy]`` adds ``lxml`` (needed to actually parse ``.3mf`` and
  ``.xaml``) and ``pycollada`` (needed to parse ``.dae``). SpatialGeometry's own
  ``collision`` extra already lists ``pycollada`` directly, so COLLADA works
  here without reaching for trimesh's ``easy`` extra.
* ``trimesh[recommend]`` adds ``cascadio`` (needed to parse ``.step``/``.stp``).
* ``trimesh[deprecated]`` adds ``openctm`` (needed to parse ``.ctm``, a
  legacy format trimesh itself is phasing out).

In short: trimesh's extras are only worth installing here if you need
collision geometry from a ``.3mf``/``.step``/``.stp``/``.xaml``/``.ctm`` file
and don't care about displaying it in Swift (which can't load any of those
anyway) -- for the display-and-collision workflow this page is about, they're
not needed.

The overlap -- the formats that work for both display **and**
collision -- is:

* **STL** (``.stl``)
* **OBJ** (``.obj``)
* **PLY** (``.ply``)
* **COLLADA** (``.dae``)
* **glTF / GLB** (``.gltf`` / ``.glb``)

These five are what the rest of this page focuses on. If you're not sure
where to start, jump straight to the :ref:`summary table <fileformats-table>`.


The five formats
=================

STL
---

STL ("stereolithography") is the oldest format here by some margin --
3D Systems introduced it in 1987 for early 3D-printing hardware, and it has
barely changed since. It stores nothing but a flat, unindexed list of
triangles (each with its own three vertices and a normal) -- no materials,
no hierarchy, no scene structure. That simplicity is exactly why it's still
everywhere: it's the default mesh export from essentially every CAD package
(SolidWorks, Fusion 360, OnShape, ...), which makes it the most common
format you'll encounter for robot link and end-effector geometry exported
from CAD, and it shows up constantly as the ``visual``/``collision`` mesh
reference in URDF files. It comes in a compact binary form and a verbose
ASCII form. Color has no place in the original spec; some tools (`the
"Magics" convention
<https://en.wikipedia.org/wiki/STL_(file_format)#Color_in_binary_STL>`_)
squeeze an RGB triplet into unused bytes of binary STL, which both trimesh
and Swift's loader recognize, but it's a convention, not a standard, and
plenty of files omit it. STL isn't going away -- it's too deeply embedded
in CAD/3D-printing tooling -- but it isn't gaining new capabilities either.

OBJ
---

OBJ dates to the late 1980s/early 1990s, from Wavefront Technologies'
Advanced Visualizer -- one of the earliest widely-shared 3D interchange
formats, and still one of the simplest to read or write by hand. Geometry
lives in a plain-text ``.obj`` file; materials (including texture image
references) live in a companion ``.mtl`` file that the ``.obj`` points to,
so an OBJ mesh is really a small bundle of files, not one. A common,
widely-supported (though never formally standardized) extension appends an
RGB triplet directly to each vertex line, and both trimesh and Swift's
``OBJLoader`` understand it. OBJ's ubiquity means it remains a safe,
boring, universally-supported choice for a single static textured mesh --
neither growing nor shrinking, it's the dependable middle ground.

PLY
---

PLY (the "Stanford Polygon" format) was created in 1994 at Stanford for 3D
scanning research -- it's the format behind the famous Stanford Bunny
dataset. Unlike STL and OBJ, per-vertex and per-face color is a first-class
part of the spec, not a bolt-on convention, which made PLY the natural
choice for scanned or photogrammetry data where color comes from the scan
itself rather than a painted material. It supports both compact binary and
readable ASCII encodings. Texture-mapping support was added later and is
less universally implemented than vertex color. PLY remains the format of
choice whenever per-vertex color matters more than materials/textures --
point clouds, scan output, mesh-processing research -- and continues to see
steady use in that niche.

COLLADA (.dae)
--------------

COLLADA ("COLLAborative Design Activity") is an XML-based scene-interchange
format, originally developed by Sony for game production in 2004 and later
adopted by the Khronos Group. It's considerably richer than STL/OBJ/PLY --
full scene graphs, named materials with texture references, skinning and
animation -- which also makes it more verbose and slower to parse. It's the
other format (alongside STL) you'll most often meet in robotics: many ROS
robot description packages ship their visual meshes as ``.dae`` precisely
because it carries per-link color/material information that STL can't.
COLLADA's influence peaked in the mid-2000s to mid-2010s (SketchUp, early
Blender pipelines); Khronos itself now positions `glTF <https://www.khronos.org/gltf/>`_
as COLLADA's successor for new work, and COLLADA is best understood today as
a well-supported legacy format rather than one to reach for by choice.

External textures
^^^^^^^^^^^^^^^^^^

A ``.dae`` file is plain XML text -- it doesn't embed image data inline. Its
``<library_images>`` block just points at texture files by relative path
(e.g. ``textures/diffuse.jpg``), so a textured COLLADA model is really a
small *bundle*: the ``.dae`` plus one or more JPEG/PNG files sitting
alongside it, the same idea as OBJ's separate ``.mtl`` (see below) except the
material *definitions* live inline in the XML and only the *images* are
external.

Treat the ``.dae`` and its texture files as one unit -- never move or rename
one without the other. The relative paths are resolved against wherever the
``.dae`` itself was loaded from. For collision, trimesh's COLLADA loader
defaults to ``ignore_broken=True``, so a missing or unreachable texture
image doesn't stop it extracting the geometry Coal needs -- textures are
irrelevant to collision anyway.

Other formats have a version of this same "bundle" problem: OBJ's geometry
(``.obj``) and materials (``.mtl``, itself referencing texture images) are
always two separate files, never optional. glTF has it too, but only in the
plain-text ``.gltf`` form, which can reference a separate ``.bin`` geometry
buffer and separate image files -- ``.glb`` packs geometry, buffers, and
images into one file, sidestepping the issue entirely, and is generally
preferable for that reason when you control the export. STL and PLY are
normally single, self-contained files with no companion assets, so this
doesn't apply to them.

How a renderer handles a missing companion file (texture, ``.mtl``, ``.bin``)
is a renderer-specific concern, not something SpatialGeometry itself is
involved in -- see `Swift's documentation <https://jhavl.github.io/swift/>`_
for that.

glTF / GLB
----------

glTF ("GL Transmission Format") is Khronos's 2017 answer to "what should
COLLADA have been" -- often described as the JPEG of 3D. It was designed
from the outset for fast runtime loading: its layout maps almost directly
onto GPU vertex/index buffers, so there's very little parsing work between
"bytes on disk" and "triangles on screen". It supports PBR materials,
textures, per-vertex color, skinning, and animation. It comes in two forms:
plain-text ``.gltf`` (JSON, typically referencing separate ``.bin`` geometry
and image files -- multiple files again, like OBJ) and single-file binary
``.glb``, which packs geometry, materials, and textures into one file. glTF
is the actively-growing format of this group -- it's three.js's own
preferred format, Blender's default export target, and the format most new
web/AR/VR/game tooling is designed around. If you're generating meshes
fresh rather than reusing existing CAD/URDF assets, glTF/GLB is generally
the best default.


.. _fileformats-table:

Summary table
==============

.. list-table::
   :header-rows: 1
   :widths: 18 14 16 14 14 14

   * - Format
     - Vertex color
     - Materials / face color
     - Texture (UV)
     - Single file
     - Binary form
   * - `STL <https://en.wikipedia.org/wiki/STL_(file_format)>`__
     - ~ (non-standard)
     - ✗
     - ✗
     - ✓
     - ✓ (or ASCII)
   * - `OBJ <https://en.wikipedia.org/wiki/Wavefront_.obj_file>`__
     - ~ (non-standard)
     - ✓ (via ``.mtl``)
     - ✓ (via ``.mtl``)
     - ✗ (+ ``.mtl`` + images)
     - ✗ (text only)
   * - `PLY <https://en.wikipedia.org/wiki/PLY_(file_format)>`__
     - ✓
     - ✓
     - ~ (common, not core)
     - ✓ (usually)
     - ✓ (or ASCII)
   * - `COLLADA <https://www.khronos.org/collada/>`__
     - ✓
     - ✓
     - ✓ (external images)
     - ✗ (+ image files)
     - ✗ (XML text)
   * - `glTF/GLB <https://www.khronos.org/gltf/>`__
     - ✓
     - ✓ (PBR)
     - ✓
     - ✓ (``.glb``) / ✗ (``.gltf``)
     - ✓ (``.glb``)

**Rule of thumb:** reusing a CAD/URDF export -- use whatever it already is
(almost always STL or COLLADA, both fully supported). Generating a mesh
fresh, or need color/texture with a single self-contained file -- use GLB.
Working with scanned/point-cloud data where per-vertex color matters most --
use PLY.
