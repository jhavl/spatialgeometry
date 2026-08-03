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

    def test_mesh_collision_false(self):
        s0 = gm.Mesh("test.stl", collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

    def test_mesh2(self):
        s0 = gm.Mesh("test.stl")

        ans = {
            "stype": "mesh",
            "scale": [1.0, 1.0, 1.0],
            "filename": "test.stl",
            "t": [0.0, 0.0, 0.0],
            "q": [0.0, 0.0, 0.0, 1],
            "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "color": 5000268,
            "opacity": 1.0,
        }

        self.assertEqual(s0.to_dict(), ans)

    def test_cylinder(self):
        s0 = gm.Cylinder(1, 1, collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

    def test_sphere(self):
        s0 = gm.Sphere(1, collision=False)
        with self.assertRaises(ValueError):
            s0._init_coal()

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


if __name__ == "__main__":  # pragma nocover
    unittest.main()
