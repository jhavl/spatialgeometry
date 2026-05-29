#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as nt
from spatialmath.base import r2q

from spatialgeometry.scene import (
    node_init,
    node_update,
    scene_graph_children,
    scene_graph_tree,
)


def _make_T(tx=0.0, ty=0.0, tz=0.0, yaw=0.0):
    c = np.cos(yaw)
    s = np.sin(yaw)
    T = np.eye(4, dtype=np.float64, order="F")
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[:3, 3] = [tx, ty, tz]
    return T


class TestSceneBackend(unittest.TestCase):
    def test_scene_graph_tree_propagates_from_root(self):
        root_T = _make_T(tx=1.0, yaw=0.2)
        child_T = _make_T(ty=2.0, yaw=-0.1)

        root_wT = np.eye(4, dtype=np.float64, order="F")
        child_wT = np.eye(4, dtype=np.float64, order="F")
        root_wq = np.zeros(4, dtype=np.float64)
        child_wq = np.zeros(4, dtype=np.float64)

        root = node_init(0, root_T, root_wT, root_wq, None, [])
        child = node_init(0, child_T, child_wT, child_wq, root, [])
        node_update(root, 1, None, [child])

        # Tree propagation should always start from the top-most ancestor.
        scene_graph_tree(child)

        expected_root_wT = root_T
        expected_child_wT = root_T @ child_T

        nt.assert_allclose(root_wT, expected_root_wT)
        nt.assert_allclose(child_wT, expected_child_wT)
        nt.assert_allclose(root_wq, r2q(expected_root_wT[:3, :3], order="xyzs"), atol=1e-12)
        nt.assert_allclose(child_wq, r2q(expected_child_wT[:3, :3], order="xyzs"), atol=1e-12)

    def test_scene_graph_children_uses_node_as_local_root(self):
        root_T = _make_T(tx=3.0, yaw=0.5)
        child_T = _make_T(ty=1.0, yaw=0.3)
        grandchild_T = _make_T(tz=2.0, yaw=-0.2)

        root_wT = np.eye(4, dtype=np.float64, order="F")
        child_wT = np.eye(4, dtype=np.float64, order="F")
        grandchild_wT = np.eye(4, dtype=np.float64, order="F")
        root_wq = np.zeros(4, dtype=np.float64)
        child_wq = np.zeros(4, dtype=np.float64)
        grandchild_wq = np.zeros(4, dtype=np.float64)

        root = node_init(0, root_T, root_wT, root_wq, None, [])
        child = node_init(0, child_T, child_wT, child_wq, root, [])
        grandchild = node_init(0, grandchild_T, grandchild_wT, grandchild_wq, child, [])

        node_update(root, 1, None, [child])
        node_update(child, 1, root, [grandchild])

        # Children propagation intentionally does not walk to parent first.
        scene_graph_children(child)

        expected_child_wT = child_T
        expected_grandchild_wT = child_T @ grandchild_T

        nt.assert_allclose(child_wT, expected_child_wT)
        nt.assert_allclose(grandchild_wT, expected_grandchild_wT)
        nt.assert_allclose(child_wq, r2q(expected_child_wT[:3, :3], order="xyzs"), atol=1e-12)
        nt.assert_allclose(
            grandchild_wq,
            r2q(expected_grandchild_wT[:3, :3], order="xyzs"),
            atol=1e-12,
        )

    def test_node_update_reparents_child(self):
        root_T = _make_T(tx=1.0)
        child_T = _make_T(ty=1.0)
        parent2_T = _make_T(tz=1.0)

        root_wT = np.eye(4, dtype=np.float64, order="F")
        child_wT = np.eye(4, dtype=np.float64, order="F")
        parent2_wT = np.eye(4, dtype=np.float64, order="F")
        root_wq = np.zeros(4, dtype=np.float64)
        child_wq = np.zeros(4, dtype=np.float64)
        parent2_wq = np.zeros(4, dtype=np.float64)

        root = node_init(0, root_T, root_wT, root_wq, None, [])
        child = node_init(0, child_T, child_wT, child_wq, root, [])
        parent2 = node_init(0, parent2_T, parent2_wT, parent2_wq, None, [])

        node_update(root, 1, None, [child])
        scene_graph_tree(child)
        nt.assert_allclose(child_wT, root_T @ child_T)

        node_update(parent2, 1, None, [child])
        node_update(child, 0, parent2, [])

        scene_graph_tree(child)
        nt.assert_allclose(child_wT, parent2_T @ child_T)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
