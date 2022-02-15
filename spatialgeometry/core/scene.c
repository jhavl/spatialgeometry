/**
 * \file scene.c
 * \author Jesse Haviland
 *
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "scene.h"
#include <stdio.h>

// forward defines
static PyObject *scene_graph_tree(PyObject *self, PyObject *args);
static PyObject *scene_graph_children(PyObject *self, PyObject *args);
static PyObject *node_update(PyObject *self, PyObject *args);
static PyObject *node_init(PyObject *self, PyObject *args);

int _check_array_type(PyObject *toCheck);
void propogate_T(Node *node, npy_float64 *parent_wT);

// void A(Link *link, npy_float64 *ret, double eta);
void mult(npy_float64 *A, npy_float64 *B, npy_float64 *C);
void copy(npy_float64 *A, npy_float64 *B);
void _eye(npy_float64 *data);
void _inv(npy_float64 *m, npy_float64 *invOut);
void _r2q(npy_float64 *r, npy_float64 *q);
void _cross(npy_float64 *a, npy_float64 *b, npy_float64 *ret, int n);

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

    propogate_T(node, (npy_float64 *)NULL);

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

    propogate_T(node, (npy_float64 *)NULL);

    Py_RETURN_NONE;
}

static PyObject *node_update(PyObject *self, PyObject *args)
{
    Node *node, *parent, *child;
    PyObject *py_node, *py_parent, *py_children, *ret;
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
            !(child = (Node *)PyIter_Next(iter_children)))
            return NULL;

        node->children[i] = (Node *)PyCapsule_GetPointer(child, "Node");
    }

    Py_DECREF(iter_children);

    Py_RETURN_NONE;
}

static PyObject *node_init(PyObject *self, PyObject *args)
{
    Node *node, *parent, *child;
    PyObject *py_parent, *py_children, *ret;
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
    node->wT = (npy_float64 *)PyArray_DATA(py_wT);
    node->wq = (npy_float64 *)PyArray_DATA(py_wq);

    // Allocate children array
    node->children = (Node **)PyMem_RawCalloc(node->n_children, sizeof(Node *));

    // Set shape pointers
    iter_children = PyObject_GetIter(py_children);

    for (int i = 0; i < node->n_children; i++)
    {
        if (
            !(child = (Node *)PyIter_Next(iter_children)))
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

void propogate_T(Node *node, npy_float64 *parent_wT)
{
    if (parent_wT == NULL)
    {
        // We have the top node
        copy(node->T, node->wT);
        _r2q(node->wT, node->wq);
    }
    else
    {
        mult(parent_wT, node->T, node->wT);
        _r2q(node->wT, node->wq);
    }

    for (int i = 0; i < node->n_children; i++)
    {
        propogate_T(node->children[i], node->wT);
    }
}

// void A(Link *link, npy_float64 *ret, double eta)
// {
//     npy_float64 *v;

//     if (!link->isjoint)
//     {
//         copy(link->A, ret);
//         return;
//     }

//     if (link->isflip)
//     {
//         eta = -eta;
//     }

//     // Calculate the variable part of the link
//     v = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
//     link->op(v, eta);

//     // Multiply ret = A * v
//     mult(link->A, v, ret);
//     free(v);
// }

void copy(npy_float64 *A, npy_float64 *B)
{
    // copy A into B
    B[0] = A[0];
    B[1] = A[1];
    B[2] = A[2];
    B[3] = A[3];
    B[4] = A[4];
    B[5] = A[5];
    B[6] = A[6];
    B[7] = A[7];
    B[8] = A[8];
    B[9] = A[9];
    B[10] = A[10];
    B[11] = A[11];
    B[12] = A[12];
    B[13] = A[13];
    B[14] = A[14];
    B[15] = A[15];
}

void mult(npy_float64 *A, npy_float64 *B, npy_float64 *C)
{
    const int N = 4;
    int i, j, k;
    double num;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            num = 0;
            for (k = 0; k < N; k++)
            {
                num += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = num;
        }
    }
}

void _eye(npy_float64 *data)
{
    data[0] = 1;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
    data[4] = 0;
    data[5] = 1;
    data[6] = 0;
    data[7] = 0;
    data[8] = 0;
    data[9] = 0;
    data[10] = 1;
    data[11] = 0;
    data[12] = 0;
    data[13] = 0;
    data[14] = 0;
    data[15] = 1;
}

void _inv(npy_float64 *m, npy_float64 *inv)
{
    inv[0] = m[0];
    inv[1] = m[4];
    inv[2] = m[8];

    inv[4] = m[1];
    inv[5] = m[5];
    inv[6] = m[9];

    inv[8] = m[2];
    inv[9] = m[6];
    inv[10] = m[10];

    inv[3] = -(inv[0] * m[3] + inv[1] * m[7] + inv[2] * m[11]);
    inv[7] = -(inv[4] * m[3] + inv[5] * m[7] + inv[6] * m[11]);
    inv[11] = -(inv[8] * m[3] + inv[9] * m[7] + inv[10] * m[11]);

    inv[12] = 0;
    inv[13] = 0;
    inv[14] = 0;
    inv[15] = 1;
}

void _r2q(npy_float64 *r, npy_float64 *q)
{
    double t12p, t13p, t23p;
    double t12m, t13m, t23m;
    double d1, d2, d3, d4;

    t12p = pow((r[0 * 4 + 1] + r[1 * 4 + 0]), 2);
    t13p = pow((r[0 * 4 + 2] + r[2 * 4 + 0]), 2);
    t23p = pow((r[1 * 4 + 2] + r[2 * 4 + 1]), 2);

    t12m = pow((r[0 * 4 + 1] - r[1 * 4 + 0]), 2);
    t13m = pow((r[0 * 4 + 2] - r[2 * 4 + 0]), 2);
    t23m = pow((r[1 * 4 + 2] - r[2 * 4 + 1]), 2);

    d1 = pow((r[0 * 4 + 0] + r[1 * 4 + 1] + r[2 * 4 + 2] + 1), 2);
    d2 = pow((r[0 * 4 + 0] - r[1 * 4 + 1] - r[2 * 4 + 2] + 1), 2);
    d3 = pow((-r[0 * 4 + 0] + r[1 * 4 + 1] - r[2 * 4 + 2] + 1), 2);
    d4 = pow((-r[0 * 4 + 0] - r[1 * 4 + 1] + r[2 * 4 + 2] + 1), 2);

    q[3] = sqrt(d1 + t23m + t13m + t12m) / 4.0;
    q[0] = sqrt(t23m + d2 + t12p + t13p) / 4.0;
    q[1] = sqrt(t13m + t12p + d3 + t23p) / 4.0;
    q[2] = sqrt(t12m + t13p + t23p + d4) / 4.0;

    // transfer sign from rotation element differences
    if (r[2 * 4 + 1] < r[1 * 4 + 2])
        q[0] = -q[0];
    if (r[0 * 4 + 2] < r[2 * 4 + 0])
        q[1] = -q[1];
    if (r[1 * 4 + 0] < r[0 * 4 + 1])
        q[2] = -q[2];
}

void _cross(npy_float64 *a, npy_float64 *b, npy_float64 *ret, int n)
{
    ret[0] = a[1 * n] * b[2 * n] - a[2 * n] * b[1 * n];
    ret[1 * n] = a[2 * n] * b[0] - a[0] * b[2 * n];
    ret[2 * n] = a[0] * b[1 * n] - a[1 * n] * b[0];
    // ret[0] = b[0 * n];
    // ret[1 * n] = b[1 * n];
    // ret[2 * n] = b[2 * n];
}
