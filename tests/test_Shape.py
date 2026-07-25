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
        s0 = gm.Shape()
        s0.wT = np.eye(4)
        nt.assert_almost_equal(np.eye(4), s0.wT)

    def test_collision_shape_wt(self):
        s0 = gm.CollisionShape()
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


if __name__ == "__main__":  # pragma nocover
    unittest.main()
