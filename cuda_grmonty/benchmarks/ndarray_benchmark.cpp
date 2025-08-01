/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_grmonty/ndarray.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <random>
#include <tuple>
#include <vector>

constexpr int N = 1024;

std::vector<std::tuple<int, int>> generate_random_indices(int size, unsigned seed = 123);

static void BM_NDArrayElementAccess(benchmark::State &state) {
    ndarray::NDArray<int, 2> a({N, N});

    std::vector<std::tuple<int, int>> indices = generate_random_indices(N);

    for (auto _ : state) {
        for (auto const &index : indices) {
            const auto [i, j] = index;
            int x = a(i, j);
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }
}

static void BM_StdArrayElementAccess(benchmark::State &state) {
    std::array<std::array<int, N>, N> a;

    std::vector<std::tuple<int, int>> indices = generate_random_indices(N);

    for (auto _ : state) {
        for (auto const &index : indices) {
            const auto [i, j] = index;
            int x = a[i][j];
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }
}

static void BM_StdVectorElementAccess(benchmark::State &state) {
    std::vector<std::vector<int>> a;
    a.resize(N, std::vector<int>(N));

    std::vector<std::tuple<int, int>> indices = generate_random_indices(N);

    for (auto _ : state) {
        for (auto const &index : indices) {
            const auto [i, j] = index;
            int x = a[i][j];
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }
}

static void BM_CStyleArrayElementAccess(benchmark::State &state) {
    auto a = new int *[N];
    for (int i = 0; i < N; ++i) {
        a[i] = new int[N];
    }

    std::vector<std::tuple<int, int>> indices = generate_random_indices(N);

    for (auto _ : state) {
        for (auto const &index : indices) {
            const auto [i, j] = index;
            int x = a[i][j];
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }

    for (int i = 0; i < N; ++i) {
        delete[] a[i];
    }

    delete[] a;
}

std::vector<std::tuple<int, int>> generate_random_indices(int size, unsigned seed) {
    static std::mt19937 rng(seed);
    static std::uniform_int_distribution<size_t> dist(0, size - 1);

    std::vector<std::tuple<int, int>> indices;

    for (int i = 0; i < size; ++i) {
        indices.emplace_back(dist(rng), dist(rng));
    }
    return indices;
}

BENCHMARK(BM_NDArrayElementAccess);
BENCHMARK(BM_StdArrayElementAccess);
BENCHMARK(BM_StdVectorElementAccess);
BENCHMARK(BM_CStyleArrayElementAccess);

BENCHMARK_MAIN();
