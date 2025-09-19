/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdio>

namespace utils {

/**
 * @brief Macro to check CUDA API calls for errors.
 *
 * @param ans CUDA API call to check.
 */
#define gpuErrchk(ans)                               \
    {                                                \
        utils::gpuAssert((ans), __FILE__, __LINE__); \
    }

/**
 * @brief Check CUDA return code and print error message if needed.
 *
 * @param code  CUDA return code to check.
 * @param file  Source file where the check is performed.
 * @param line  Line number where the check is performed.
 * @param abort If true, aborts the program on error (default: true).
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

}; /* namespace utils */
