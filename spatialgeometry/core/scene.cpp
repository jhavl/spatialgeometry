/**
 * \file scene.cpp
 * \author Jesse Haviland
 *
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "scene.h"
#include "linalg.h"

static PyMethodDef sceneMethods[] = {
    {"scene_graph_tree",
     (PyCFunction)scene_graph_tree,
     METH_VARARGS,
     "Link"},
    {"scene_graph_children",
     (PyCFunction)scene_graph_children,
     METH_VARARGS,
     "Link"},
    {"node_update",
     (PyCFunction)node_update,
     METH_VARARGS,
     "Link"},
    {"node_init",
     (PyCFunction)node_init,
     METH_VARARGS,
     "Link"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef scenemodule =
    {
        PyModuleDef_HEAD_INIT,
        "scene",
        "Scene Graph",
        -1,
        sceneMethods};

PyMODINIT_FUNC PyInit_scene(void)
{
    import_array();
    return PyModule_Create(&scenemodule);
}

extern "C"
{

    static PyObject *scene_graph_tree(PyObject *self, PyObject *args)
    {
        Node *node;
        PyObject *py_node;

        if (!PyArg_ParseTuple(args, "O",
                              &py_node))
            return NULL;

        // Get existing note pointer
        if (!(node = (Node *)PyCapsule_GetPointer(py_node, "Node")))
        {
            return NULL;
        }

        // Get to parent node
        while (1)
        {
            if (node->parent != NULL)
            {
                node = node->parent;
            }
            else
            {
                break;
            }
        }

        propogate_T(node, (npy_float64 *)NULL, (MapMatrix4dc)NULL);

        Py_RETURN_NONE;
    }

    static PyObject *scene_graph_children(PyObject *self, PyObject *args)
    {
        Node *node;
        PyObject *py_node;

        if (!PyArg_ParseTuple(args, "O",
                              &py_node))
            return NULL;

        // Get existing node pointer
        if (!(node = (Node *)PyCapsule_GetPointer(py_node, "Node")))
        {
            return NULL;
        }

        propogate_T(node, (npy_float64 *)NULL, (MapMatrix4dc)NULL);

        Py_RETURN_NONE;
    }

    static PyObject *node_update(PyObject *self, PyObject *args)
    {
        Node *node, *parent;
        PyObject *py_node, *py_parent, *py_children, *ret, *child;
        PyObject *iter_children;
        int n_children;

        if (!PyArg_ParseTuple(args, "OiOO",
                              &py_node,
                              &n_children,
                              &py_parent,
                              &py_children))
            return NULL;

        // Get existing node pointer
        if (!(node = (Node *)PyCapsule_GetPointer(py_node, "Node")))
        {
            return NULL;
        }

        // Check the parent of the node
        if (py_parent == Py_None)
        {
            node->parent = NULL;
        }
        else if (!(parent = (Node *)PyCapsule_GetPointer(py_parent, "Node")))
        {
            return NULL;
        }
        else
        {
            node->parent = parent;
        }

        // Set the number of children
        node->n_children = n_children;

        // Allocate children array
        node->children = (Node **)PyMem_RawCalloc(node->n_children, sizeof(Node));

        // Set shape pointers
        iter_children = PyObject_GetIter(py_children);

        for (int i = 0; i < node->n_children; i++)
        {
            if (
                !(child = (PyObject *)PyIter_Next(iter_children)))
                return NULL;

            node->children[i] = (Node *)PyCapsule_GetPointer(child, "Node");
        }

        Py_DECREF(iter_children);

        Py_RETURN_NONE;
    }

    static PyObject *node_init(PyObject *self, PyObject *args)
    {
        Node *node, *parent;
        PyObject *py_parent, *py_children, *ret, *child;
        PyObject *iter_children;
        PyArrayObject *py_T, *py_wT, *py_wq;

        node = (Node *)PyMem_RawMalloc(sizeof(Node));

        if (!PyArg_ParseTuple(args, "iO!O!O!OO",
                              &node->n_children,
                              &PyArray_Type, &py_T,
                              &PyArray_Type, &py_wT,
                              &PyArray_Type, &py_wq,
                              &py_parent,
                              &py_children))
            return NULL;

        // Check the parent of the node
        if (py_parent == Py_None)
        {
            node->parent = NULL;
        }
        else if (!(parent = (Node *)PyCapsule_GetPointer(py_parent, "Node")))
        {
            return NULL;
        }
        else
        {
            node->parent = parent;
        }

        // Set the transform arrays
        node->T = (npy_float64 *)PyArray_DATA(py_T);
        new (&node->Tm) MapMatrix4dc(node->T);

        node->wT = (npy_float64 *)PyArray_DATA(py_wT);
        new (&node->wTm) MapMatrix4dc(node->wT);

        node->wq = (npy_float64 *)PyArray_DATA(py_wq);
        new (&node->wqv) MapVector4(node->wq);

        // Allocate children array
        node->children = (Node **)PyMem_RawCalloc(node->n_children, sizeof(Node *));

        // Set shape pointers
        iter_children = PyObject_GetIter(py_children);

        for (int i = 0; i < node->n_children; i++)
        {
            if (
                !(child = (PyObject *)PyIter_Next(iter_children)))
                return NULL;

            node->children[i] = (Node *)PyCapsule_GetPointer(child, "Node");
        }

        Py_DECREF(iter_children);

        ret = PyCapsule_New(node, "Node", NULL);
        return ret;
    }

    /* ----------------------------------------------------------------- */
    // Private Methods
    /* ----------------------------------------------------------------- */

    int _check_array_type(PyObject *toCheck)
    {
        PyArray_Descr *desc;

        desc = PyArray_DescrFromObject(toCheck, NULL);

        // Check if desc is a number or a sympy symbol
        if (!PyDataType_ISNUMBER(desc))
        {
            PyErr_SetString(PyExc_TypeError, "Symbolic value");
            return 0;
        }

        return 1;
    }

    void propogate_T(Node *node, npy_float64 *parent_wT, MapMatrix4dc parent_wTm)
    {
        if (parent_wT == NULL)
        {
            // We have the top node
            node->wTm = node->Tm;
            r2q(node->wTm, node->wq);
        }
        else
        {
            // mult(parent_wT, node->T, node->wT);
            node->wTm = parent_wTm * node->Tm;
            r2q(node->wTm, node->wq);
        }

        for (int i = 0; i < node->n_children; i++)
        {
            propogate_T(node->children[i], node->wT, node->wTm);
        }
    }

    void r2q(MapMatrix4dc r, npy_float64 *q)
    {
        // float tr = r[0] + r[1 * 4 + 1] + r(2, + 2];

        // if (tr > 0)
        // {
        //     float S = sqrt(tr + 1.0) * 2; // S=4*qw
        //     q[3] = 0.25 * S;
        //     q[0] = (r(2, + 1] - r[1 * 4 + 2]) / S;
        //     q[1] = (r(0, + 2] - r[2 * 4 + 0]) / S;
        //     q[2] = (r[1 * 4 + 0] - r[0 * 4 + 1]) / S;
        // }
        // else if ((r[0 * 4 + 0] > r[1 * 4 + 1]) & (r[0 * 4 + 0] > r[2 * 4 + 2]))
        // {
        //     float S = sqrt(1.0 + r[0 * 4 + 0] - r[1 * 4 + 1] - r[2 * 4 + 2]) * 2; // S=4*q[0]
        //     q[3] = (r[2 * 4 + 1] - r[1 * 4 + 2]) / S;
        //     q[0] = 0.25 * S;
        //     q[1] = (r[0 * 4 + 1] + r[1 * 4 + 0]) / S;
        //     q[2] = (r[0 * 4 + 2] + r[2 * 4 + 0]) / S;
        // }
        // else if (r[1 * 4 + 1] > r[2 * 4 + 2])
        // {
        //     float S = sqrt(1.0 + r[1 * 4 + 1] - r[0 * 4 + 0] - r[2 * 4 + 2]) * 2; // S=4*q[1]
        //     q[3] = (r[0 * 4 + 2] - r[2 * 4 + 0]) / S;
        //     q[0] = (r[0 * 4 + 1] + r[1 * 4 + 0]) / S;
        //     q[1] = 0.25 * S;
        //     q[2] = (r[1 * 4 + 2] + r[2 * 4 + 1]) / S;
        // }
        // else
        // {
        //     float S = sqrt(1.0 + r[2 * 4 + 2] - r[0 * 4 + 0] - r[1 * 4 + 1]) * 2; // S=4*q[2]
        //     q[3] = (r[1 * 4 + 0] - r[0 * 4 + 1]) / S;
        //     q[0] = (r[0 * 4 + 2] + r[2 * 4 + 0]) / S;
        //     q[1] = (r[1 * 4 + 2] + r[2 * 4 + 1]) / S;
        //     q[2] = 0.25 * S;
        // }

        float tr = r(0, 0) + r(1, 1) + r(2, 2);

        if (tr > 0)
        {
            float S = sqrt(tr + 1.0) * 2; // S=4*qw
            q[3] = 0.25 * S;
            q[0] = (r(2, 1) - r(1, 2)) / S;
            q[1] = (r(0, 2) - r(2, 0)) / S;
            q[2] = (r(1, 0) - r(0, 1)) / S;
        }
        else if ((r(0, 0) > r(1, 1)) & (r(0, 0) > r(2, 2)))
        {
            float S = sqrt(1.0 + r(0, 0) - r(1, 1) - r(2, 2)) * 2; // S=4*q[0]
            q[3] = (r(2, 1) - r(1, 2)) / S;
            q[0] = 0.25 * S;
            q[1] = (r(0, 1) + r(1, 0)) / S;
            q[2] = (r(0, 2) + r(2, 0)) / S;
        }
        else if (r(1, 1) > r(2, 2))
        {
            float S = sqrt(1.0 + r(1, 1) - r(0, 0) - r(2, 2)) * 2; // S=4*q[1]
            q[3] = (r(0, 2) - r(2, 0)) / S;
            q[0] = (r(0, 1) + r(1, 0)) / S;
            q[1] = 0.25 * S;
            q[2] = (r(1, 2) + r(2, 1)) / S;
        }
        else
        {
            float S = sqrt(1.0 + r(2, 2) - r(0, 0) - r(1, 1)) * 2; // S=4*q[2]
            q[3] = (r(1, 0) - r(0, 1)) / S;
            q[0] = (r(0, 2) + r(2, 0)) / S;
            q[1] = (r(1, 2) + r(2, 1)) / S;
            q[2] = 0.25 * S;
        }
    }

} /* extern "C" */