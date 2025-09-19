/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>
#include <utility>

#include "cuda_grmonty/consts.hpp"

namespace linalg {

namespace matrix {

/**
 * @brief Compute the determinant of a 1x1 matrix.
 *
 * @tparam T     Scalar type of the matrix (e.g., int, float, double).
 * @param matrix Pointer to the first element of the matrix.
 *
 * @return Determinant of the 1x1 matrix.
 */
template <typename T>
static T det_1x1(T *matrix) {
    return matrix[0];
}

/**
 * @brief Compute the determinant of a 2x2 matrix.
 *
 * @tparam T     Scalar type of the matrix.
 * @param matrix Pointer to the first element of the matrix, stored in row-major order.
 *
 * @return Determinant of the 2x2 matrix.
 */
template <typename T>
static T det_2x2(T *matrix) {
    return (matrix[0] * matrix[3] - matrix[1] * matrix[2]);
}

/**
 * @brief Compute the determinant of a 3x3 matrix.
 *
 * @tparam T     Scalar type of the matrix.
 * @param matrix Pointer to the first element of the matrix, stored in row-major order.
 *
 * @return Determinant of the 3x3 matrix.
 */
template <typename T>
static T det_3x3(T *matrix) {
    /* clang-format off */
    return (
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
      - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
      + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
    );
    /* clang-format on */
}

/**
 * @brief Compute the determinant of a 4x4 matrix.
 *
 * @tparam T     Scalar type of the matrix.
 * @param matrix Pointer to the first element of the matrix, stored in row-major order.
 *
 * @return Determinant of the 4x4 matrix.
 */
template <typename T>
static T det_4x4(T *matrix) {
    /* clang-format off */
    return (
        matrix[0] * (
            matrix[5] * (matrix[10] * matrix[15] - matrix[11] * matrix[14])
          - matrix[6] * (matrix[9] * matrix[15] - matrix[11] * matrix[13])
          + matrix[7] * (matrix[9] * matrix[14] - matrix[10] * matrix[13])
        )
      - matrix[1] * (
            matrix[4] * (matrix[10] * matrix[15] - matrix[11] * matrix[14])
          - matrix[6] * (matrix[8] * matrix[15] - matrix[11] * matrix[12])
          + matrix[7] * (matrix[8] * matrix[14] - matrix[10] * matrix[12])
        )
      + matrix[2] * (
            matrix[4] * (matrix[9] * matrix[15] - matrix[11] * matrix[13])
          - matrix[5] * (matrix[8] * matrix[15] - matrix[11] * matrix[12])
          + matrix[7] * (matrix[8] * matrix[13] - matrix[9] * matrix[12])
        )
      - matrix[3] * (
            matrix[4] * (matrix[9] * matrix[14] - matrix[10] * matrix[13])
          - matrix[5] * (matrix[8] * matrix[14] - matrix[10] * matrix[12])
          + matrix[6] * (matrix[8] * matrix[13] - matrix[9] * matrix[12])
        )
    );
    /* clang-format on */
}

/**
 * @brief Compute the determinant of an n×n matrix (floating-point specialization).
 *
 * @tparam T     Floating-point type (float, double).
 * @param n      Dimension of the square matrix.
 * @param matrix Pointer to the first element of the matrix, stored in row-major order.
 *
 * @return Determinant of the n×n matrix as a floating-point value.
 */
template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value, T>::type det_nxn_fp(const int n, T *matrix) {
    int sign = 1;
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        T max_val = std::abs(matrix[i * n + i]);
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(matrix[i * n + j]) > max_val) {
                max_val = std::abs(matrix[i * n + j]);
                pivot = j;
            }
        }

        /* singular matrix */
        if (std::abs(matrix[pivot * n + i]) < consts::eps) {
            return 0;
        }

        /* swap rows */
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                std::swap(matrix[i * n + j], matrix[pivot * n + j]);
            }
            sign *= -1;
        }

        /* reduce */
        for (int j = i + 1; j < n; ++j) {
            matrix[j * n + i] /= matrix[i * n + i];
            for (int k = i + 1; k < n; ++k) {
                matrix[j * n + k] -= matrix[j * n + i] * matrix[i * n + k];
            }
        }
    }

    T result = sign;
    for (int i = 0; i < n; ++i) {
        result *= matrix[i * n + i];
    }

    return result;
}

/**
 * @brief Compute the determinant of an n×n matrix (integer specialization).
 *
 * @tparam T     Integral type (int, long, etc.).
 * @param n      Dimension of the square matrix.
 * @param matrix Pointer to the first element of the matrix, stored in row-major order.
 *
 * @return Determinant of the n×n matrix as an integer value.
 */
template <typename T>
static typename std::enable_if<!std::is_floating_point<T>::value, T>::type det_nxn_int(const int n, T *matrix) {
    int sign = 1;
    T prev_pivot = 1;

    for (int i = 0; i < n; ++i) {
        int pivot_row = -1;
        for (int j = i; j < n; ++j) {
            if (std::abs(matrix[i * n + j]) != 0) {
                pivot_row = j;
                break;
            }
        }

        /* singular matrix */
        if (pivot_row == -1) {
            return 0;
        }

        /* swap rows */
        if (pivot_row != i) {
            for (int j = 0; j < n; ++j) {
                std::swap(matrix[i * n + j], matrix[pivot_row * n + j]);
            }
            sign *= -1;
        }

        T pivot = matrix[i * n + i];

        /* reduce */
        for (int j = i + 1; j < n; ++j) {
            for (int k = i + 1; k < n; ++k) {
                matrix[j * n + k] = (matrix[j * n + k] * pivot - matrix[j * n + i] * matrix[i * n + k]) / prev_pivot;
            }
        }

        prev_pivot = pivot;
    }

    return sign * matrix[n * n - 1];
}

/**
 * @brief Dispatch determinant calculation for an n×n matrix.
 *
 * Selects specialized functions for 1×1 through 4×4 matrices. For larger matrices, dispatches to either floating-point
 * or integer general determinant functions based on the type T.
 *
 * @tparam T     Scalar type of the matrix.
 * @param n      Dimension of the square matrix.
 * @param matrix Pointer to the first element of the matrix, stored in row-major order.
 *
 * @return Determinant of the n×n matrix.
 */
template <typename T>
T det(int n, T *matrix) {
    if (n == 1) {
        return det_1x1(matrix);
    } else if (n == 2) {
        return det_2x2(matrix);
    } else if (n == 3) {
        return det_3x3(matrix);
    } else if (n == 4) {
        return det_4x4(matrix);
    } else if constexpr (std::is_floating_point_v<T>) {
        return det_nxn_fp(n, matrix);
    } else {
        return det_nxn_int(n, matrix);
    }
}

}; /* namespace matrix */

}; /* namespace linalg */
