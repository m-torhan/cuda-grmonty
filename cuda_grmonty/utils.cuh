/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdio>

namespace utils {

#define gpuErrchk(ans)                               \
    {                                                \
        utils::gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

}; /* namespace utils */
