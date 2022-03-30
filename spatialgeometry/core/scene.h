/**
 * \file scene.h
 * \author Jesse Haviland
 * \brief Definitions for c file
 *
 */

#ifndef _SCENE_H_
#define _SCENE_H_

#include <math.h>
#include <numpy/arrayobject.h>
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    typedef struct Node Node;

    struct Node
    {
        /**********************************************************
         *************** kinematic parameters *********************
         **********************************************************/
        npy_float64 *T;  // static transform
        npy_float64 *wT; // world transform
        npy_float64 *wq; // world quaternion

        MapMatrix4dc Tm;
        MapMatrix4dc wTm;
        MapVector4 wqv;

        int n_children;
        Node *parent;
        Node **children;
    };

    static PyObject *scene_graph_tree(PyObject *self, PyObject *args);
    static PyObject *scene_graph_children(PyObject *self, PyObject *args);
    static PyObject *node_update(PyObject *self, PyObject *args);
    static PyObject *node_init(PyObject *self, PyObject *args);

    int _check_array_type(PyObject *toCheck);
    void propogate_T(Node *node, npy_float64 *parent_wT, MapMatrix4dc parent_wTm);

    void r2q(MapMatrix4dc r, npy_float64 *q);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif