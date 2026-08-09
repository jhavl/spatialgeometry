#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import unittest
import spatialmath as sm
import spatialgeometry as gm

from tests import skip_no_collision_checking


class TestShape(unittest.TestCase):
    def test_init(self):
        gm.Cuboid([1, 1, 1], base=sm.SE3(0, 0, 0))
        gm.Cylinder(1, 1, base=sm.SE3(2, 0, 0))
        gm.Sphere(1, base=sm.SE3(4, 0, 0))

    def test_color_setter_formats(self):
        shape = gm.Cuboid([1, 1, 1], base=sm.SE3(0, 0, 0))

        shape.color = [0.1, 0.2, 0.3]

        self.assertEqual(shape.color[0], 0.1)
        self.assertEqual(shape.color[1], 0.2)
        self.assertEqual(shape.color[2], 0.3)
        self.assertEqual(shape.color[3], 1)

        shape.color = [0.1, 0.2, 0.3, 0.5]

        self.assertEqual(shape.color[0], 0.1)
        self.assertEqual(shape.color[1], 0.2)
        self.assertEqual(shape.color[2], 0.3)
        self.assertEqual(shape.color[3], 0.5)

        shape.color = (0.1, 0.2, 0.3)

        self.assertEqual(shape.color[0], 0.1)
        self.assertEqual(shape.color[1], 0.2)
        self.assertEqual(shape.color[2], 0.3)
        self.assertEqual(shape.color[3], 1)

        shape.color = (100, 200, 250, 100)

        self.assertAlmostEqual(shape.color[0], 100 / 255)
        self.assertAlmostEqual(shape.color[1], 200 / 255)
        self.assertAlmostEqual(shape.color[2], 250 / 255)
        self.assertEqual(shape.color[3], 100 / 255)

    def test_color_setter_all_integer_input_stays_json_serializable(self):
        # Regression test: color=[1, 0, 0, 1] (a natural way to write
        # "opaque red") used to leave self._color as numpy int64 -- nothing
        # here needs 0-255 normalisation (all values <= 1), so the only
        # thing that made 3-length int input safe by accident was
        # concatenate()'s promotion when appending the alpha default; a
        # fully-specified 4-length int color skipped that entirely.
        # self.color[3] (opacity) being int64 broke real (non-mocked)
        # json.dumps() sends in SwiftRoute.py, uncaught by any existing
        # test since they're all FakeBrowser-mocked.
        import json

        shape = gm.Cuboid([1, 1, 1], color=[1, 0, 0, 1])

        self.assertIsInstance(shape.color[3], float)
        json.dumps(shape.to_dict())  # must not raise TypeError

    def test_opacity_property(self):
        shape = gm.Cuboid([1, 1, 1], color=[0.1, 0.2, 0.3, 0.5])
        self.assertEqual(shape.opacity, 0.5)

        shape.opacity = 0.25
        self.assertEqual(shape.opacity, 0.25)
        # rgb untouched by an opacity-only set
        self.assertEqual(shape.color[0], 0.1)
        self.assertEqual(shape.color[1], 0.2)
        self.assertEqual(shape.color[2], 0.3)

        shape.opacity = 100  # >1 -- auto-normalised as 0-255 input
        self.assertAlmostEqual(shape.opacity, 100 / 255)

    def test_set_alpha_warns_and_matches_opacity(self):
        shape = gm.Cuboid([1, 1, 1], color=[0.1, 0.2, 0.3, 1.0])
        with self.assertWarns(FutureWarning):
            shape.set_alpha(0.4)
        self.assertEqual(shape.opacity, 0.4)

    def test_to_dict(self):
        s1 = gm.Cylinder(1, 1)

        ans = {
            "stype": "cylinder",
            "radius": 1.0,
            "length": 1.0,
            "t": [0.0, 0.0, 0.0],
            "q": [0, 0, 0, 1.0],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
        }

        self.assertEqual(s1.to_dict()["stype"], ans["stype"])
        self.assertEqual(s1.to_dict()["v"], ans["v"])
        self.assertEqual(s1.to_dict()["color"], ans["color"])
        nt.assert_almost_equal(s1.to_dict()["q"], ans["q"])

    def test_to_dict2(self):
        s1 = gm.Sphere(1)

        ans = {
            "stype": "sphere",
            "radius": 1.0,
            "t": [0.0, 0.0, 0.0],
            "q": [0, 0.0, 0.0, 1.0],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
        }

        self.assertEqual(s1.to_dict(), ans)

    def test_fk_dict(self):
        s1 = gm.Cylinder(1, 1)

        ans = {
            "t": [0.0, 0.0, 0.0],
            "q": [0, 0, 0, 1.0],
        }

        nt.assert_almost_equal(s1.fk_dict()["t"], ans["t"])
        nt.assert_almost_equal(s1.fk_dict()["q"], ans["q"])

    def test_fk_dict2(self):
        s1 = gm.Sphere(1)

        ans = {"t": [0.0, 0.0, 0.0], "q": [0, 0, 0, 1]}

        self.assertEqual(s1.fk_dict(), ans)

    def test_wq_matches_spatialmath_r2q(self):
        # fk_dict()'s "q" comes from _wq, populated by the C++ r2q() in
        # scene_nb.cpp (Shepperd's method) every _propogate_scene_tree()
        # call -- a different algorithm to spatialmath.base.r2q's Cayley's
        # method, and never cross-checked against it before. Covers the
        # cases quaternion-extraction methods most commonly get wrong:
        # identity, near-identity, and 90/180/179.99 degrees about each axis.
        cases = {
            "identity": sm.SE3(),
            "near_identity": sm.SE3.Rx(0.001, unit="deg"),
            "rx_90": sm.SE3.Rx(90, unit="deg"),
            "ry_90": sm.SE3.Ry(90, unit="deg"),
            "rz_90": sm.SE3.Rz(90, unit="deg"),
            "rx_180": sm.SE3.Rx(180, unit="deg"),
            "ry_180": sm.SE3.Ry(180, unit="deg"),
            "rz_180": sm.SE3.Rz(180, unit="deg"),
            "rx_179.99": sm.SE3.Rx(179.99, unit="deg"),
            "arbitrary1": sm.SE3.Rx(37) * sm.SE3.Ry(-52) * sm.SE3.Rz(101),
            "arbitrary2": sm.SE3.Rx(-170) * sm.SE3.Ry(88) * sm.SE3.Rz(-45),
        }

        for name, T in cases.items():
            shape = gm.Cuboid([0.1, 0.1, 0.1], pose=T)
            shape._propogate_scene_tree()

            expected = sm.base.r2q(T.R, order="xyzs")

            # q and -q represent the same rotation -- accept either sign.
            same = np.allclose(shape._wq, expected, atol=1e-9)
            opposite = np.allclose(shape._wq, -np.array(expected), atol=1e-9)
            self.assertTrue(same or opposite, msg=f"{name}: {shape._wq} vs {expected}")

    @skip_no_collision_checking
    def test_collision(self):
        s0 = gm.Cuboid([1, 1, 1], base=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], base=sm.SE3(0.5, 0, 0))
        s2 = gm.Cuboid([1, 1, 1], base=sm.SE3(3, 0, 0))

        s0._propogate_scene_children()
        s1._propogate_scene_children()
        s2._propogate_scene_children()

        c0 = s0.iscollided(s1)
        c1 = s0.iscollided(s2)

        self.assertTrue(c0)
        self.assertFalse(c1)

    def test_wt(self):
        s0 = gm.Cuboid([1, 1, 1], base=sm.SE3(0, 0, 0))
        s0.wT = np.eye(4)

    def test_color(self):
        s0 = gm.Sphere(1, color="red")
        self.assertEqual(s0.color, (1.0, 0.0, 0.0, 1.0))

    def test_color2(self):
        s0 = gm.Sphere(1, color="sdgfsg")
        self.assertEqual(s0.color, (0.95, 0.5, 0.25, 1.0))

    def test_color3(self):
        s0 = gm.Sphere(1, color=[255, 255, 255])
        self.assertEqual(s0.color, (1.0, 1.0, 1.0, 1.0))

    def test_shape_wt(self):
        s0 = gm.Axes(1.0)
        s0.wT = np.eye(4)
        nt.assert_almost_equal(np.eye(4), s0.wT)

    def test_collision_shape_wt(self):
        s0 = gm.Cuboid([1, 1, 1])
        s0.wT = np.eye(4)
        nt.assert_almost_equal(np.eye(4), s0.wT)

    def test_set_T_on_parented_shape(self):
        # Regression test: setting T on a shape that already has a
        # scene_parent used to raise AttributeError -- SceneNode._T's
        # setter referenced self.parent.wT, neither of which exist.
        parent = gm.Cuboid([1, 1, 1], pose=sm.SE3.Trans(1, 0, 0))
        child = gm.Cuboid([1, 1, 1])
        parent.attach(child)

        child.T = sm.SE3.Trans(0, 2, 0)

        expected = sm.SE3.Trans(1, 0, 0).A @ sm.SE3.Trans(0, 2, 0).A
        nt.assert_almost_equal(child._wT, expected)

    def test_scene_parent_rejects_self_parenting(self):
        a = gm.Cuboid([1, 1, 1])
        with self.assertRaises(ValueError):
            a.scene_parent = a

    def test_scene_parent_rejects_two_node_cycle(self):
        a = gm.Cuboid([1, 1, 1])
        b = gm.Cuboid([1, 1, 1])
        b.scene_parent = a
        with self.assertRaises(ValueError):
            a.scene_parent = b

    def test_scene_parent_rejects_longer_cycle(self):
        x = gm.Cuboid([1, 1, 1])
        y = gm.Cuboid([1, 1, 1])
        z = gm.Cuboid([1, 1, 1])
        y.scene_parent = x
        z.scene_parent = y
        with self.assertRaises(ValueError):
            x.scene_parent = z

    def test_attach_to_rejects_cycle(self):
        # attach_to() goes through scene_parent -- same protection applies.
        m = gm.Cuboid([1, 1, 1])
        n = gm.Cuboid([1, 1, 1])
        n.attach_to(m)
        with self.assertRaises(ValueError):
            m.attach_to(n)

    def test_valid_reparenting_still_works(self):
        p = gm.Cuboid([1, 1, 1])
        q = gm.Cuboid([1, 1, 1])
        q.scene_parent = p
        self.assertIs(q.scene_parent, p)

    def test_mesh_collision_false(self):
        s0 = gm.Mesh("test.stl", collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

    def test_mesh_scalar_scale(self):
        s0 = gm.Mesh("test.stl", scale=2.0)
        nt.assert_almost_equal(s0.scale, [2.0, 2.0, 2.0])

    def test_mesh_list_scale_still_works(self):
        s0 = gm.Mesh("test.stl", scale=[1.0, 2.0, 3.0])
        nt.assert_almost_equal(s0.scale, [1.0, 2.0, 3.0])

    def test_mesh2(self):
        s0 = gm.Mesh("test.stl")

        ans = {
            "stype": "mesh",
            "scale": [1.0, 1.0, 1.0],
            "y_up": False,
            "filename": "test.stl",
            "t": [0.0, 0.0, 0.0],
            "q": [0.0, 0.0, 0.0, 1],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
            "use_vertex_colors": True,
        }

        self.assertEqual(s0.to_dict(), ans)

    def test_mesh_use_vertex_colors(self):
        # No explicit color -- defer to whatever's baked into the file.
        s0 = gm.Mesh("test.stl")
        self.assertTrue(s0.to_dict()["use_vertex_colors"])

        # Explicit color at construction -- always overrides.
        s1 = gm.Mesh("test.stl", color=[1.0, 0.0, 0.0, 1.0])
        self.assertFalse(s1.to_dict()["use_vertex_colors"])

        # Explicit color set after construction -- also overrides.
        s2 = gm.Mesh("test.stl")
        s2.color = [0.0, 1.0, 0.0, 1.0]
        self.assertFalse(s2.to_dict()["use_vertex_colors"])

    def test_cylinder(self):
        s0 = gm.Cylinder(1, 1, collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

    def test_sphere(self):
        s0 = gm.Sphere(1, collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

    def test_ellipsoid(self):
        s0 = gm.Ellipsoid([1, 1, 1], collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

    def test_ellipsoid2(self):
        s0 = gm.Ellipsoid([1, 1, 1])

        ans = {
            "stype": "ellipsoid",
            "radii": [1.0, 1.0, 1.0],
            "t": [0.0, 0.0, 0.0],
            "q": [0.0, 0.0, 0.0, 1],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
        }

        self.assertEqual(s0.to_dict(), ans)

    def test_Cuboid(self):
        s0 = gm.Cuboid(None, collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

    def test_Cuboid2(self):
        s0 = gm.Cuboid([1, 1, 1])

        ans = {
            "stype": "cuboid",
            "scale": [1.0, 1.0, 1.0],
            "t": [0.0, 0.0, 0.0],
            "q": [0.0, 0.0, 0.0, 1],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
        }

        self.assertEqual(s0.to_dict(), ans)

    def test_Axes_defaults(self):
        s0 = gm.Axes(1.0)

        ans = {
            "stype": "axes",
            "t": [0.0, 0.0, 0.0],
            "q": [0.0, 0.0, 0.0, 1],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
            "length": 1.0,
            "arrows": False,
            "radius": 0.0,
            "linewidth": 1.0,
        }

        self.assertEqual(s0.to_dict(), ans)

    def test_Axes_arrows_passes_through_radius_and_linewidth(self):
        s0 = gm.Axes(2.0, arrows=True, radius=0.05, linewidth=3.0)

        d = s0.to_dict()
        self.assertTrue(d["arrows"])
        self.assertEqual(d["radius"], 0.05)
        self.assertEqual(d["linewidth"], 3.0)

    def test_Arrow_defaults(self):
        s0 = gm.Arrow(1.0)

        ans = {
            "stype": "arrow",
            "t": [0.0, 0.0, 0.0],
            "q": [0.0, 0.0, 0.0, 1],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
            "length": 1.0,
            "radius": 0.0,
            "linewidth": 1.0,
            "head_length": 0.2,
            "head_radius": 0.2,
        }

        self.assertEqual(s0.to_dict(), ans)

    def test_Arrow_radius_and_linewidth_are_independent_params(self):
        # radius and linewidth are mutually exclusive at render time (see
        # shapes.js in swift), but both are always accepted/stored here --
        # it's the renderer's job to pick which one applies, not this class's.
        s0 = gm.Arrow(1.0, radius=0.1, linewidth=5.0)

        d = s0.to_dict()
        self.assertEqual(d["radius"], 0.1)
        self.assertEqual(d["linewidth"], 5.0)

    def test_repr_distinguishes_deprecated_alias_from_its_base(self):
        # Box is a deprecated alias of Cuboid sharing the same stype --
        # repr must still tell them apart (it used to show "cuboid" for
        # both, making them indistinguishable).
        cuboid = gm.Cuboid([1, 2, 3])
        box = gm.Box([1, 2, 3])

        self.assertTrue(repr(cuboid).startswith("Cuboid("))
        self.assertTrue(repr(box).startswith("Box("))
        self.assertNotEqual(repr(cuboid), repr(box))

    def test_repr_is_single_line_and_shows_constructor_params(self):
        s0 = gm.Cylinder(1.0, 2.0)
        r = repr(s0)

        self.assertNotIn("\n", r)
        self.assertEqual(
            r,
            "Cylinder(radius=1.0, length=2.0, color=(0.3, 0.3, 0.3), "
            "pose='t = 0, 0, 0; rpy/zyx = 0°, 0°, 0°')",
        )

    def test_repr_shows_color_always_opacity_only_if_not_1(self):
        opaque = gm.Cuboid([1, 1, 1], color=[1, 0, 0, 1])
        self.assertIn("color=(1.0, 0.0, 0.0)", repr(opaque))
        self.assertNotIn("opacity=", repr(opaque))

        transparent = gm.Cuboid([1, 1, 1], color=[1, 0, 0, 0.5])
        self.assertIn("color=(1.0, 0.0, 0.0)", repr(transparent))
        self.assertIn("opacity=0.5", repr(transparent))

    def test_repr_color_is_plain_float_not_numpy_scalar(self):
        # self._color's elements can be numpy.float64 after color=[...]
        # with a list/array input -- repr() must show a plain "1.0", not
        # numpy's own "np.float64(1.0)".
        s0 = gm.Sphere(1.0, color=[1, 0, 0, 1])
        self.assertNotIn("np.float64", repr(s0))

    def test_repr_handles_all_integer_3_tuple_color(self):
        # color=(1, 0, 0) -- all-integer, 3-length (no alpha), a natural
        # way to write opaque red. Must not be misread as 0-255 range
        # (only normalises if any component > 1.0), must pad alpha to
        # 1.0 (so opacity is correctly omitted), and must not leak
        # numpy scalars into the repr.
        s0 = gm.Cuboid([1, 1, 1], color=(1, 0, 0))
        r = repr(s0)
        self.assertIn("color=(1.0, 0.0, 0.0)", r)
        self.assertNotIn("opacity=", r)
        self.assertNotIn("np.float64", r)

    def test_str_is_single_line(self):
        s0 = gm.Cuboid([1, 1, 1], pose=sm.SE3.Trans(1, 2, 3))
        s = str(s0)

        self.assertNotIn("\n", s)
        self.assertEqual(s, "Cuboid at t = 1, 2, 3; rpy/zyx = 0°, 0°, 0°")

    def test_scene_group_repr_and_str(self):
        group = gm.SceneGroup()
        group.append(gm.Sphere(1.0))

        self.assertEqual(repr(group), f"SceneGroup([{gm.Sphere(1.0)!r}])")
        self.assertTrue(str(group).startswith("SceneGroup at "))

    def test_scene_group_constructor_accepts_initial_elements(self):
        cube = gm.Cuboid([1, 1, 1])
        sphere = gm.Sphere(1.0)
        group = gm.SceneGroup([cube, sphere])

        self.assertEqual(len(group), 2)
        self.assertIs(cube.scene_parent, group)
        self.assertIs(sphere.scene_parent, group)

    def test_scene_group_mutators_set_and_clear_scene_parent(self):
        group = gm.SceneGroup()

        box = gm.Cuboid([1, 1, 1])
        group.append(box)
        self.assertIs(box.scene_parent, group)

        extra = gm.Sphere(0.5)
        group.extend([extra])
        self.assertIs(extra.scene_parent, group)

        mid = gm.Sphere(0.3)
        group.insert(1, mid)
        self.assertEqual(list(group), [box, mid, extra])
        self.assertIs(mid.scene_parent, group)

        group.remove(box)
        self.assertIsNone(box.scene_parent)
        self.assertEqual(len(group), 2)

        popped = group.pop()
        self.assertIs(popped, extra)
        self.assertIsNone(popped.scene_parent)

        group.clear()
        self.assertEqual(len(group), 0)

    def test_scene_group_setitem_and_delitem_set_and_clear_scene_parent(self):
        group = gm.SceneGroup([gm.Cuboid([1, 1, 1]), gm.Sphere(1.0)])
        old_item = group[0]
        new_item = gm.Sphere(2.0)

        group[0] = new_item
        self.assertIs(group[0], new_item)
        self.assertIs(new_item.scene_parent, group)
        self.assertIsNone(old_item.scene_parent)

        del group[0]
        self.assertEqual(len(group), 1)

    def test_scene_group_scene_parent_does_not_flatten_its_elements(self):
        # Setting a SceneGroup's own scene_parent should only affect the
        # group itself -- its own elements must stay its elements, not get
        # reparented to the new parent too.
        parent = gm.Cuboid([1, 1, 1])
        group = gm.SceneGroup([gm.Cuboid([1, 1, 1])])

        group.scene_parent = parent
        self.assertIs(group.scene_parent, parent)
        self.assertEqual(len(group), 1)

        parent.T = sm.SE3(5, 0, 0)
        parent._propogate_scene_tree()
        nt.assert_almost_equal(group._wT[:3, 3], [5, 0, 0])
        nt.assert_almost_equal(group[0]._wT[:3, 3], [5, 0, 0])

    def test_Path_defaults(self):
        # points is accepted/stored as 3xN (this ecosystem's own point-set
        # convention), but the wire format transposes to a flat N x 3 list
        # of [x, y, z] waypoints -- the natural shape for the JS side to
        # build one THREE.Vector3 per point.
        points = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        s0 = gm.Path(points)

        ans = {
            "stype": "path",
            "t": [0.0, 0.0, 0.0],
            "q": [0.0, 0.0, 0.0, 1],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
            "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "radius": 0.0,
            "linewidth": 1.0,
        }

        self.assertEqual(s0.to_dict(), ans)

    def test_Path_radius_and_linewidth_are_independent_params(self):
        points = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        s0 = gm.Path(points, radius=0.05, linewidth=3.0)

        d = s0.to_dict()
        self.assertEqual(d["radius"], 0.05)
        self.assertEqual(d["linewidth"], 3.0)

    def test_Path_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            gm.Path(np.zeros((2, 3)))  # not 3xN

    def test_Path_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            gm.Path(np.zeros((3, 1)))  # a single point isn't a polyline

    def test_Path_accepts_plain_list_input(self):
        # points is documented as ArrayLike, not just ndarray.
        s0 = gm.Path([[0, 1], [0, 0], [0, 0]])
        self.assertEqual(s0.to_dict()["points"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


class TestSceneTreePrint(unittest.TestCase):
    def test_leaf_tree_children_is_a_single_line(self):
        leaf = gm.Sphere(1.0)
        self.assertEqual(leaf.tree_children(), repr(leaf))

    def test_tree_children_shows_only_this_subtree_not_siblings(self):
        parent = gm.Cuboid([1, 1, 1])
        child = gm.Sphere(1.0)
        sibling = gm.Cylinder(2.0, 3.0)  # distinguishable repr from child
        child.scene_parent = parent
        sibling.scene_parent = parent

        # From the child's own perspective, tree_children() must not walk
        # back up through its parent (matching _propogate_scene_children()'s
        # "not through parents" scope) or sideways to its sibling.
        result = child.tree_children()
        self.assertEqual(result, repr(child))
        self.assertNotIn(repr(sibling), result)

    def test_tree_children_indentation_reflects_depth(self):
        root = gm.Cuboid([1, 1, 1])
        mid = gm.Sphere(1.0)
        leaf = gm.Cylinder(1.0, 1.0)
        mid.scene_parent = root
        leaf.scene_parent = mid

        lines = root.tree_children().split("\n")
        self.assertEqual(lines, [repr(root), "    " + repr(mid), "        " + repr(leaf)])

    def test_tree_walks_to_root_regardless_of_which_node_it_is_called_on(self):
        root = gm.Cuboid([1, 1, 1])
        mid = gm.Sphere(1.0)
        leaf = gm.Cylinder(1.0, 1.0)
        mid.scene_parent = root
        leaf.scene_parent = mid

        # Calling .tree() from the leaf must produce the same whole-tree
        # rendering as calling it from the root (minus the "<==" marker,
        # checked separately below).
        from_root = root.tree().replace("  <==", "")
        from_leaf = leaf.tree().replace("  <==", "")
        self.assertEqual(from_root, from_leaf)

    def test_tree_marks_the_calling_node(self):
        root = gm.Cuboid([1, 1, 1])
        leaf = gm.Sphere(1.0)
        leaf.scene_parent = root

        result = leaf.tree()
        lines = result.split("\n")
        self.assertFalse(lines[0].endswith("<=="))  # root, not the caller
        self.assertTrue(lines[1].endswith("<=="))   # leaf, the caller

    def test_tree_on_an_unparented_node_is_just_itself(self):
        alone = gm.Sphere(1.0)
        self.assertEqual(alone.tree(), repr(alone) + "  <==")

    def test_scene_group_needs_no_special_casing(self):
        # A SceneGroup's list elements and its scene-graph children are the
        # same underlying list -- the walker needs no SceneGroup-specific
        # branch, it just recurses into scene_children like any other node.
        anchor = gm.Cuboid([1, 1, 1])
        s0 = gm.Sphere(1.0)
        s1 = gm.Cylinder(1.0, 1.0)
        group = gm.SceneGroup([s0, s1])
        group.scene_parent = anchor

        result = anchor.tree_children()
        self.assertIn(repr(s0), result)
        self.assertIn(repr(s1), result)


class TestBoundingBox(unittest.TestCase):
    def test_cuboid_local_corners_bounds_extents(self):
        s0 = gm.Cuboid([1, 2, 3])
        self.assertEqual(s0.corners().shape, (3, 8))
        nt.assert_almost_equal(s0.bounds(), [[-0.5, 0.5], [-1, 1], [-1.5, 1.5]])
        nt.assert_almost_equal(s0.extents(), [1, 2, 3])

    def test_sphere_local_extents(self):
        s0 = gm.Sphere(2.0)
        nt.assert_almost_equal(s0.extents(), [4, 4, 4])

    def test_cylinder_local_extents(self):
        # radius=1, length=4 -- axis along Z (see Cylinder's own docstring)
        s0 = gm.Cylinder(1.0, 4.0)
        nt.assert_almost_equal(s0.extents(), [2, 2, 4])

    def test_local_extents_are_pose_independent(self):
        # Rotating a shape must not change its own local extents -- that's
        # the whole distinction between world=False (default) and
        # world=True.
        s0 = gm.Cuboid([1, 2, 3], pose=sm.SE3.Rz(45, unit="deg"))
        nt.assert_almost_equal(s0.extents(), [1, 2, 3])

    def test_world_extents_reflect_current_pose(self):
        # A 1x2x3 cuboid rotated 45 degrees about Z has a larger axis-
        # aligned footprint in x/y, unchanged in z.
        s0 = gm.Cuboid([1, 2, 3], pose=sm.SE3.Rz(45, unit="deg"))
        world_extents = s0.extents(world=True)
        self.assertGreater(world_extents[0], 1)
        self.assertGreater(world_extents[1], 2)
        nt.assert_almost_equal(world_extents[2], 3)

    def test_world_corners_shape(self):
        s0 = gm.Cuboid([1, 2, 3], pose=sm.SE3.Rz(45, unit="deg") * sm.SE3.Trans(5, 0, 0))
        self.assertEqual(s0.corners(world=True).shape, (3, 8))

    def test_unimplemented_shape_raises_not_implemented(self):
        # Axes/Arrow/Path don't implement _local_corners() yet -- corners()
        # must fail loudly, not silently return a wrong value.
        s0 = gm.Axes(1.0)
        with self.assertRaises(NotImplementedError):
            s0.corners()

    def test_mesh_extents(self):
        import tempfile
        import trimesh

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/asym.stl"
            trimesh.creation.box(extents=[1, 2, 3]).export(path)

            s0 = gm.Mesh(path)
            nt.assert_almost_equal(s0.extents(), [1, 2, 3], decimal=6)

            s1 = gm.Mesh(path, scale=[2, 2, 2])
            nt.assert_almost_equal(s1.extents(), [2, 4, 6], decimal=6)

    def test_mesh_extents_independent_of_collision_flag(self):
        # Unlike _init_coal(), the bounding box is a plain geometric fact
        # and must work even when collision=False.
        import tempfile
        import trimesh

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/box.stl"
            trimesh.creation.box(extents=[1, 1, 1]).export(path)

            s0 = gm.Mesh(path, collision=False)
            nt.assert_almost_equal(s0.extents(), [1, 1, 1], decimal=6)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
