/**
 * scene_nb.cpp — nanobind port of scene.cpp
 *
 * Python glue rewritten: NB_MODULE + nb::class_ replace PyMethodDef /
 * PyModuleDef / PyCapsule; typed function parameters replace
 * PyArg_ParseTuple. Node is now a proper C++ object owned via
 * std::unique_ptr -- no more leak from PyCapsule's NULL destructor
 * (the original never freed a single Node).
 *
 * Math unchanged from scene.cpp: wT = parent_wT * T propagated down the
 * scene tree, plus a rotation-matrix-to-quaternion extraction (r2q).
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <cmath>

namespace nb = nanobind;

using Matrix4dc = Eigen::Matrix4d;
using MapMatrix4dc = Eigen::Map<Matrix4dc>;
using MapVector4 = Eigen::Map<Eigen::Vector4d>;

// Unconstrained, like fknm_nb.cpp's `darray` -- callers (SceneNode.py)
// always pass F-contiguous (column-major) 4x4 arrays, matched directly
// onto Eigen's default column-major Matrix4d storage with no copy. Not
// validated here, same implicit contract the original C code relied on.
using darray = nb::ndarray<double>;

struct Node {
    double *T;
    double *wT;
    double *wq;

    MapMatrix4dc Tm;
    MapMatrix4dc wTm;
    MapVector4 wqv;

    Node *parent = nullptr;
    std::vector<Node *> children;

    Node(double *T_, double *wT_, double *wq_)
        : T(T_), wT(wT_), wq(wq_), Tm(T_), wTm(wT_), wqv(wq_) {}
};

static void r2q(const MapMatrix4dc &r, double *q) {
    double tr = r(0, 0) + r(1, 1) + r(2, 2);

    if (tr > 0) {
        double S = std::sqrt(tr + 1.0) * 2.0; // S=4*qw
        q[3] = 0.25 * S;
        q[0] = (r(2, 1) - r(1, 2)) / S;
        q[1] = (r(0, 2) - r(2, 0)) / S;
        q[2] = (r(1, 0) - r(0, 1)) / S;
    } else if ((r(0, 0) > r(1, 1)) && (r(0, 0) > r(2, 2))) {
        double S = std::sqrt(1.0 + r(0, 0) - r(1, 1) - r(2, 2)) * 2.0;
        q[3] = (r(2, 1) - r(1, 2)) / S;
        q[0] = 0.25 * S;
        q[1] = (r(0, 1) + r(1, 0)) / S;
        q[2] = (r(0, 2) + r(2, 0)) / S;
    } else if (r(1, 1) > r(2, 2)) {
        double S = std::sqrt(1.0 + r(1, 1) - r(0, 0) - r(2, 2)) * 2.0;
        q[3] = (r(0, 2) - r(2, 0)) / S;
        q[0] = (r(0, 1) + r(1, 0)) / S;
        q[1] = 0.25 * S;
        q[2] = (r(1, 2) + r(2, 1)) / S;
    } else {
        double S = std::sqrt(1.0 + r(2, 2) - r(0, 0) - r(1, 1)) * 2.0;
        q[3] = (r(1, 0) - r(0, 1)) / S;
        q[0] = (r(0, 2) + r(2, 0)) / S;
        q[1] = (r(1, 2) + r(2, 1)) / S;
        q[2] = 0.25 * S;
    }
}

// parent_wTm == nullptr means "treat node as the root", matching both
// scene.cpp's propagate_T(node, NULL, NULL) and scene.py's
// _propagate_T(node, None) -- used by scene_graph_children to propagate
// only downward regardless of node's actual parent.
static void propagate_T(Node *node, const MapMatrix4dc *parent_wTm) {
    if (parent_wTm == nullptr) {
        node->wTm = node->Tm;
    } else {
        node->wTm = (*parent_wTm) * node->Tm;
    }
    r2q(node->wTm, node->wq);

    for (Node *child : node->children) {
        propagate_T(child, &node->wTm);
    }
}

static void set_children(Node *node, int n_children, Node *parent,
                          std::vector<Node *> children) {
    node->parent = parent;
    if ((int)children.size() > n_children) {
        children.resize(n_children);
    }
    node->children = std::move(children);
}

NB_MODULE(scene, m) {
    m.doc() = "Scene Graph (nanobind port)";

    nb::class_<Node>(m, "_Node")
        .def("__repr__", [](const Node &) { return "<spatialgeometry Node>"; });

    m.def(
        "node_init",
        [](int n_children, darray T, darray wT, darray wq, Node *parent,
           std::vector<Node *> children) -> std::unique_ptr<Node> {
            auto node = std::make_unique<Node>(
                (double *)T.data(), (double *)wT.data(), (double *)wq.data());
            set_children(node.get(), n_children, parent, std::move(children));
            return node;
        },
        nb::arg("n_children"), nb::arg("T"), nb::arg("wT"), nb::arg("wq"),
        nb::arg("parent").none(), nb::arg("children"));

    m.def(
        "node_update",
        [](Node *node, int n_children, Node *parent,
           std::vector<Node *> children) {
            set_children(node, n_children, parent, std::move(children));
        },
        nb::arg("node"), nb::arg("n_children"), nb::arg("parent").none(),
        nb::arg("children"));

    m.def("scene_graph_children",
          [](Node *node) { propagate_T(node, nullptr); });

    m.def("scene_graph_tree", [](Node *node) {
        while (node->parent != nullptr) {
            node = node->parent;
        }
        propagate_T(node, nullptr);
    });
}
