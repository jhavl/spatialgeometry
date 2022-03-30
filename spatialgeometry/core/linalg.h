/**
 * \file linalg.h
 * \author Jesse Haviland
 *
 */
/* linalg.h */

#ifndef _LINALG_H_
#define _LINALG_H_

#include <Eigen/Dense>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#define Matrix4dc Eigen::Matrix4d
#define Matrix4dr Eigen::Matrix<double, 4, 4, Eigen::RowMajor>

#define MapMatrix4dc Eigen::Map<Matrix4dc>
#define MapMatrix4dr Eigen::Map<Matrix4dc>

#define MatrixJc Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::ColMajor>
#define MatrixJr Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor>
#define MapMatrixJc Eigen::Map<MatrixJc>
#define MapMatrixJr Eigen::Map<MatrixJr>

#define MatrixHc Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
#define MatrixHr Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
#define MapMatrixHc Eigen::Map<MatrixHc>
#define MapMatrixHr Eigen::Map<MatrixHr>

#define Vector3 Eigen::Vector3d
#define MapVector3 Eigen::Map<Vector3>

#define Vector4 Eigen::Vector4d
#define MapVector4 Eigen::Map<Vector4>

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif