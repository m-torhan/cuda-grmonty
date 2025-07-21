/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_grmonty/integration.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

constexpr double eps_abs = 1.0e-6;
constexpr double eps_rel = 1.0e-6;
constexpr int max_intervals = 1000;

TEST(Integration, GaussKronrodConst) {
    auto f = [](double x) { return 1.0; };
    double a = 0;
    double b = 1;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(1.0, result, eps_rel);
}

TEST(Integration, GaussKronrodLinear) {
    auto f = [](double x) { return 2.0 * x + 1.0; };
    double a = 0;
    double b = 2;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(6.0, result, eps_rel);
}

TEST(Integration, GaussKronrodSquare) {
    auto f = [](double x) { return -x * x + 1.0; };
    double a = -1;
    double b = 1;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(4.0 / 3.0, result, eps_rel);
}

TEST(Integration, GaussKronrodSin) {
    auto f = [](double x) { return std::sin(x); };
    double a = 0;
    double b = std::numbers::pi;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(2.0, result, eps_rel);
}

TEST(Integration, GaussKronrodAbs) {
    auto f = [](double x) { return std::abs(x - 0.3); };
    double a = 0;
    double b = 1;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(0.29, result, eps_rel);
}

TEST(Integration, GaussKronrodSqrt) {
    auto f = [](double x) { return std::sqrt(x); };
    double a = 0;
    double b = 1;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(2.0 / 3.0, result, eps_rel);
}

TEST(Integration, GaussKronrodLog) {
    auto f = [](double x) { return std::log(x); };
    double a = 1.0e-5;
    double b = 1;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(-0.999874870746, result, eps_rel);
}

TEST(Integration, GaussKronrodOscillations) {
    auto f = [](double x) { return std::sin(20 * x); };
    double a = 0;
    double b = std::numbers::pi;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(0.0, result, eps_rel);
}

TEST(Integration, GaussKronrodSharpPeak) {
    auto f = [](double x) { return 1.0 / (1.0 + 1000.0 * (x - 0.5) * (x - 0.5)); };
    double a = 0;
    double b = 1;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(0.0953512032278, result, eps_rel);
}

TEST(Integration, GaussKronrodStep) {
    auto f = [](double x) { return (x < 0.5) ? 0.0 : 1.0; };
    double a = 0;
    double b = 1;

    double result = integration::gauss_kronrod_61(f, a, b, eps_abs, eps_rel, max_intervals);

    ASSERT_NEAR(0.5, result, eps_rel);
}
