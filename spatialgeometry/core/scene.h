/**
 * \file scene.h
 * \author Jesse Haviland
 * \brief Definitions for c file
 *
 */

#ifndef _scene_h_
#define _scene_h_

#include <math.h>
#include <numpy/arrayobject.h>

typedef struct Node Node;

struct Node
{
    /**********************************************************
     *************** kinematic parameters *********************
     **********************************************************/
    npy_float64 *T;  // static transform
    npy_float64 *wT; // world transform
    npy_float64 *wq; // world quaternion

    int n_children;
    Node *parent;
    Node **children;
};

#endif