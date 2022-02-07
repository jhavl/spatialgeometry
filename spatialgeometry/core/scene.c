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
static PyObject *link_init(PyObject *self, PyObject *args);
static PyObject *link_update(PyObject *self, PyObject *args);

int _check_array_type(PyObject *toCheck);
// void A(Link *link, npy_float64 *ret, double eta);
void mult(npy_float64 *A, npy_float64 *B, npy_float64 *C);
void copy(npy_float64 *A, npy_float64 *B);
void _eye(npy_float64 *data);
void _inv(npy_float64 *m, npy_float64 *invOut);
void _r2q(npy_float64 *r, npy_float64 *q);
void _cross(npy_float64 *a, npy_float64 *b, npy_float64 *ret, int n);

static PyMethodDef sceneMethods[] = {
    {"link_init",
     (PyCFunction)link_init,
     METH_VARARGS,
     "Link"},
    {"link_update",
     (PyCFunction)link_update,
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

static PyObject *link_init(PyObject *self, PyObject *args)
{
    // Node *link, *parent;
    // int jointtype;
    // PyObject *ret, *py_parent;

    // PyObject *py_shape_base, *py_shape_wT, *py_shape_sT, *py_shape_sq;
    // PyObject *iter_base, *iter_wT, *iter_sT, *iter_sq;
    // PyArrayObject *pys_base, *pys_wT, *pys_sT, *pys_sq;
    // PyArrayObject *py_A, *py_fk;

    // link = (Node *)PyMem_RawMalloc(sizeof(Node));

    // if (!PyArg_ParseTuple(args, "iiiiiO!O!OOOOO",
    //                       &link->isjoint,
    //                       &link->isflip,
    //                       &jointtype,
    //                       &link->jindex,
    //                       &link->n_shapes,
    //                       &PyArray_Type, &py_A,
    //                       &PyArray_Type, &py_fk,
    //                       &py_shape_base,
    //                       &py_shape_wT,
    //                       &py_shape_sT,
    //                       &py_shape_sq,
    //                       &py_parent))
    //     return NULL;

    // if (py_parent == Py_None)
    // {
    //     parent = NULL;
    // }
    // else if (!(parent = (Link *)PyCapsule_GetPointer(py_parent, "Link")))
    // {
    //     return NULL;
    // }

    // link->A = (npy_float64 *)PyArray_DATA(py_A);
    // link->fk = (npy_float64 *)PyArray_DATA(py_fk);

    // // Set shape pointers
    // iter_base = PyObject_GetIter(py_shape_base);
    // iter_wT = PyObject_GetIter(py_shape_wT);
    // iter_sT = PyObject_GetIter(py_shape_sT);
    // iter_sq = PyObject_GetIter(py_shape_sq);

    // link->shape_base = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    // link->shape_wT = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    // link->shape_sT = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    // link->shape_sq = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));

    // for (int i = 0; i < link->n_shapes; i++)
    // {
    //     if (
    //         !(pys_base = (PyArrayObject *)PyIter_Next(iter_base)) ||
    //         !(pys_wT = (PyArrayObject *)PyIter_Next(iter_wT)) ||
    //         !(pys_sT = (PyArrayObject *)PyIter_Next(iter_sT)) ||
    //         !(pys_sq = (PyArrayObject *)PyIter_Next(iter_sq)))
    //         return NULL;

    //     link->shape_base[i] = (npy_float64 *)PyArray_DATA(pys_base);
    //     link->shape_wT[i] = (npy_float64 *)PyArray_DATA(pys_wT);
    //     link->shape_sT[i] = (npy_float64 *)PyArray_DATA(pys_sT);
    //     link->shape_sq[i] = (npy_float64 *)PyArray_DATA(pys_sq);
    // }

    // link->axis = jointtype;
    // link->parent = parent;

    // if (jointtype == 0)
    // {
    //     link->op = rx;
    // }
    // else if (jointtype == 1)
    // {
    //     link->op = ry;
    // }
    // else if (jointtype == 2)
    // {
    //     link->op = rz;
    // }
    // else if (jointtype == 3)
    // {
    //     link->op = tx;
    // }
    // else if (jointtype == 4)
    // {
    //     link->op = ty;
    // }
    // else if (jointtype == 5)
    // {
    //     link->op = tz;
    // }

    // Py_DECREF(iter_base);
    // Py_DECREF(iter_wT);
    // Py_DECREF(iter_sT);
    // Py_DECREF(iter_sq);

    // ret = PyCapsule_New(link, "Link", NULL);
    // return ret;
    Py_RETURN_NONE;
}

static PyObject *link_update(PyObject *self, PyObject *args)
{
    // Node *link, *parent;
    // int isjoint, isflip;
    // int jointtype, jindex, n_shapes;
    // PyObject *lo, *py_parent;
    // PyArrayObject *py_A, *py_fk;

    // PyObject *py_shape_base, *py_shape_wT, *py_shape_sT, *py_shape_sq;
    // PyObject *iter_base, *iter_wT, *iter_sT, *iter_sq;
    // PyArrayObject *pys_base, *pys_wT, *pys_sT, *pys_sq;

    // if (!PyArg_ParseTuple(args, "OiiiiiO!O!OOOOO",
    //                       &lo,
    //                       &isjoint,
    //                       &isflip,
    //                       &jointtype,
    //                       &jindex,
    //                       &n_shapes,
    //                       &PyArray_Type, &py_A,
    //                       &PyArray_Type, &py_fk,
    //                       &py_shape_base,
    //                       &py_shape_wT,
    //                       &py_shape_sT,
    //                       &py_shape_sq,
    //                       &py_parent))
    //     return NULL;

    // if (py_parent == Py_None)
    // {
    //     parent = NULL;
    // }
    // else if (!(parent = (Link *)PyCapsule_GetPointer(py_parent, "Link")))
    // {
    //     return NULL;
    // }

    // if (!(link = (Link *)PyCapsule_GetPointer(lo, "Link")))
    // {
    //     return NULL;
    // }

    // // Set shape pointers
    // iter_base = PyObject_GetIter(py_shape_base);
    // iter_wT = PyObject_GetIter(py_shape_wT);
    // iter_sT = PyObject_GetIter(py_shape_sT);
    // iter_sq = PyObject_GetIter(py_shape_sq);

    // if (link->shape_base != 0)
    //     free(link->shape_base);
    // if (link->shape_wT != 0)
    //     free(link->shape_wT);
    // if (link->shape_sT != 0)
    //     free(link->shape_sT);
    // if (link->shape_sq != 0)
    //     free(link->shape_sq);

    // link->shape_base = 0;
    // link->shape_wT = 0;
    // link->shape_sT = 0;
    // link->shape_sq = 0;

    // link->shape_base = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    // link->shape_wT = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    // link->shape_sT = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    // link->shape_sq = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));

    // for (int i = 0; i < n_shapes; i++)
    // {
    //     if (
    //         !(pys_base = (PyArrayObject *)PyIter_Next(iter_base)) ||
    //         !(pys_wT = (PyArrayObject *)PyIter_Next(iter_wT)) ||
    //         !(pys_sT = (PyArrayObject *)PyIter_Next(iter_sT)) ||
    //         !(pys_sq = (PyArrayObject *)PyIter_Next(iter_sq)))
    //         return NULL;

    //     link->shape_base[i] = (npy_float64 *)PyArray_DATA(pys_base);
    //     link->shape_wT[i] = (npy_float64 *)PyArray_DATA(pys_wT);
    //     link->shape_sT[i] = (npy_float64 *)PyArray_DATA(pys_sT);
    //     link->shape_sq[i] = (npy_float64 *)PyArray_DATA(pys_sq);
    // }

    // if (jointtype == 0)
    // {
    //     link->op = rx;
    // }
    // else if (jointtype == 1)
    // {
    //     link->op = ry;
    // }
    // else if (jointtype == 2)
    // {
    //     link->op = rz;
    // }
    // else if (jointtype == 3)
    // {
    //     link->op = tx;
    // }
    // else if (jointtype == 4)
    // {
    //     link->op = ty;
    // }
    // else if (jointtype == 5)
    // {
    //     link->op = tz;
    // }

    // link->isjoint = isjoint;
    // link->isflip = isflip;
    // link->A = (npy_float64 *)PyArray_DATA(py_A);
    // link->fk = (npy_float64 *)PyArray_DATA(py_fk);
    // link->jindex = jindex;
    // link->axis = jointtype;
    // link->parent = parent;
    // link->n_shapes = n_shapes;

    // Py_DECREF(iter_base);
    // Py_DECREF(iter_wT);
    // Py_DECREF(iter_sT);
    // Py_DECREF(iter_sq);

    // Py_RETURN_NONE;
    Py_RETURN_NONE;
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
