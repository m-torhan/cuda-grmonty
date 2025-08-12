/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_grmonty/ndarray.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/**
 * Tests constructor with various shapes.
 */
TEST(NDArray, Construct) { ndarray::NDArray<int, 2> a({2, 3}); }

/**
 * Tests constructor with values.
 */
TEST(NDArray, InitializeWithValues) {
    ndarray::NDArray<int, 2> a({2, 3}, {0, 1, 2, 3, 4, 5});

    ASSERT_EQ(0, a(0, 0));
    ASSERT_EQ(1, a(0, 1));
    ASSERT_EQ(2, a(0, 2));
    ASSERT_EQ(3, a(1, 0));
    ASSERT_EQ(4, a(1, 1));
    ASSERT_EQ(5, a(1, 2));
}

/**
 * Tests constructor with values.
 */
TEST(NDArray, InitializeWithValuesInvalidShape) {
    EXPECT_THROW(({ ndarray::NDArray<int, 2> a({2, 4}, {0, 1, 2, 3, 4, 5}); }), std::invalid_argument);
}

/**
 * Tests copying to uninitialized array.
 */
TEST(NDArray, CopyToUninitializedArray) {
    ndarray::NDArray<int, 2> a({2, 3}, {0, 1, 2, 3, 4, 5});
    ndarray::NDArray<int, 2> b;

    b = a;

    ASSERT_EQ(0, b(0, 0));
    ASSERT_EQ(1, b(0, 1));
    ASSERT_EQ(2, b(0, 2));
    ASSERT_EQ(3, b(1, 0));
    ASSERT_EQ(4, b(1, 1));
    ASSERT_EQ(5, b(1, 2));
}

/**
 * Tests zero dimensional array.
 */
TEST(NDArray, ZeroDim) {
    ndarray::NDArray<int, 0> a({{}});

    a = 0;

    ASSERT_EQ(0, static_cast<int>(a));
}

/**
 * Tests retrieving ndim of ndarray.
 */
TEST(NDArray, Ndim) {
    ndarray::NDArray<int, 2> a({2, 3});

    ASSERT_EQ(2, a.ndim());
}

/**
 * Tests retrieving size of ndarray.
 */
TEST(NDArray, Size) {
    std::array<int, 2> shape{2, 3};
    unsigned int size = 1;
    for (auto &s : shape) {
        size *= s;
    }

    ndarray::NDArray<int, 2> a(shape);

    ASSERT_EQ(size, a.size());
}

/**
 * Tests retrieving shape of ndarray.
 */
TEST(NDArray, Shape) {
    std::array<int, 2> shape{2, 3};

    ndarray::NDArray<int, 2> a(shape);

    std::array<int, 2> a_shape = a.shape();

    ASSERT_EQ(shape.size(), a_shape.size());
    for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
        ASSERT_EQ(shape[i], a_shape[i]);
    }
}

/**
 * Tests setting and getting single values at index.
 */
TEST(NDArray, SetGetIndex) {
    ndarray::NDArray<int, 2> a({2, 3});

    a(0, 0) = 1;

    ASSERT_EQ(1, a(0, 0));
}

/**
 * Tests assignment.
 */
TEST(NDArray, Assign) {
    ndarray::NDArray<int, 2> a({2, 3});
    ndarray::NDArray<int, 2> b({2, 3});

    a = 0;

    b = a;

    ASSERT_EQ(0, b(0, 0));
    ASSERT_EQ(0, b(0, 1));
    ASSERT_EQ(0, b(0, 2));
    ASSERT_EQ(0, b(1, 0));
    ASSERT_EQ(0, b(1, 1));
    ASSERT_EQ(0, b(1, 2));
}

/**
 * Tests assignment with invalid shape.
 */
TEST(NDArray, AssignInvalidShape) {
    ndarray::NDArray<int, 2> a({2, 3});
    ndarray::NDArray<int, 2> b({2, 4});

    a = 0;

    EXPECT_THROW({ b = a; }, std::invalid_argument);
}

/**
 * Tests converting 0-dim array to number.
 */
TEST(NDArray, ConvertToNumber) {
    ndarray::NDArray<int, 0> a({{}});

    a = 0;

    int b = a;

    ASSERT_EQ(0, b);
}

/**
 * Tests setting whole ndarray with single value.
 */
TEST(NDArray, SetWholeNDArrayWithSingleValue) {
    ndarray::NDArray<int, 2> a({2, 3});

    a = 0;
    ASSERT_EQ(0, a(0, 0));
    ASSERT_EQ(0, a(0, 1));
    ASSERT_EQ(0, a(0, 2));
    ASSERT_EQ(0, a(1, 0));
    ASSERT_EQ(0, a(1, 1));
    ASSERT_EQ(0, a(1, 2));
}

/**
 * Tests retrieveing subarray from an array.
 */
TEST(NDArray, Subarray) {
    ndarray::NDArray<int, 2> a({2, 3}, {0, 1, 2, 3, 4, 5});

    ndarray::NDArray<int, 1> b = a(0);

    ASSERT_EQ(3, b.size());
    ASSERT_EQ(1, b.ndim());
    ASSERT_EQ(3, b.shape()[0]);
    ASSERT_EQ(0, b(0));
    ASSERT_EQ(1, b(1));
    ASSERT_EQ(2, b(2));

    ndarray::NDArray<int, 1> c = a(1);

    ASSERT_EQ(3, c.size());
    ASSERT_EQ(1, c.ndim());
    ASSERT_EQ(3, c.shape()[0]);
    ASSERT_EQ(3, c(0));
    ASSERT_EQ(4, c(1));
    ASSERT_EQ(5, c(2));
}

/**
 * Tests retrieveing subarray from an array.
 */
TEST(NDArray, SubarrayMoreDims) {
    ndarray::NDArray<int, 4> a({1, 2, 1, 2}, {0, 1, 2, 3});

    ndarray::NDArray<int, 2> b = a(0, 1);

    ASSERT_EQ(2, b.size());
    ASSERT_EQ(2, b.ndim());
    ASSERT_EQ(1, b.shape()[0]);
    ASSERT_EQ(2, b.shape()[1]);
    ASSERT_EQ(2, b(0, 0));
    ASSERT_EQ(3, b(0, 1));

    ndarray::NDArray<int, 3> c = a(0);

    ASSERT_EQ(4, c.size());
    ASSERT_EQ(3, c.ndim());
    ASSERT_EQ(2, c.shape()[0]);
    ASSERT_EQ(1, c.shape()[1]);
    ASSERT_EQ(2, c.shape()[2]);
    ASSERT_EQ(0, c(0, 0, 0));
    ASSERT_EQ(1, c(0, 0, 1));
    ASSERT_EQ(2, c(1, 0, 0));
    ASSERT_EQ(3, c(1, 0, 1));
}

/**
 * Tests setting subarray.
 */
TEST(NDArray, SetSubarray) {
    ndarray::NDArray<int, 2> a({2, 3});
    ndarray::NDArray<int, 1> b({3});

    b(0) = 0;
    b(1) = 1;
    b(2) = 2;

    a(0) = b;

    ASSERT_EQ(0, a(0, 0));
    ASSERT_EQ(1, a(0, 1));
    ASSERT_EQ(2, a(0, 2));
}

/**
 * Tests setting subarray with single value.
 */
TEST(NDArray, SetSubarrayWithSingleValue) {
    ndarray::NDArray<int, 2> a({2, 3});

    a(0) = 0;

    ASSERT_EQ(0, a(0, 0));
    ASSERT_EQ(0, a(0, 1));
    ASSERT_EQ(0, a(0, 2));
}

/**
 * Tests move assignment operator.
 */
TEST(NDArray, Move) {
    ndarray::NDArray<int, 2> a({2, 3});

    a(0, 0) = 0;
    a(0, 1) = 1;
    a(0, 2) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(1, 2) = 5;

    ndarray::NDArray<int, 2> b({4, 4});
    b(0, 0) = -1;

    b = std::move(a);

    ASSERT_EQ(6, b.size());
    ASSERT_EQ(2, b.ndim());
    ASSERT_EQ(2, b.shape()[0]);
    ASSERT_EQ(3, b.shape()[1]);
    ASSERT_EQ(0, b(0, 0));
    ASSERT_EQ(1, b(0, 1));
    ASSERT_EQ(2, b(0, 2));
    ASSERT_EQ(3, b(1, 0));
    ASSERT_EQ(4, b(1, 1));
    ASSERT_EQ(5, b(1, 2));
}

/**
 * Tests reading data from stream.
 */
TEST(NDArray, ReadStream) {
    ndarray::NDArray<int, 2> a({2, 3});

    std::stringstream s;
    s << 0 << " " << 1 << " " << 2 << " " << 3 << " " << 4 << " " << 5;

    s >> a;

    ASSERT_EQ(0, a(0, 0));
    ASSERT_EQ(1, a(0, 1));
    ASSERT_EQ(2, a(0, 2));
    ASSERT_EQ(3, a(1, 0));
    ASSERT_EQ(4, a(1, 1));
    ASSERT_EQ(5, a(1, 2));
}

/**
 * Tests determinant of 1x1 integer array computation.
 */
TEST(NDArray, Det1DInt) {
    ndarray::NDArray<int, 2> a({1, 1}, {6});

    ASSERT_EQ(6, a.det());
}

/**
 * Tests determinant of 2x2 integer array computation.
 */
TEST(NDArray, Det2DInt) {
    /* clang-format off */
    ndarray::NDArray<int, 2> a({2, 2}, {3, 5, 2, 4});
    /* clang-format on */

    ASSERT_EQ(2, a.det());
}

/**
 * Tests determinant of singular 2x2 integer array computation.
 */
TEST(NDArray, Det2DIntSingular) {
    /* clang-format off */
    ndarray::NDArray<int, 2> a({2, 2}, { 3,  5,
                                      6, 10});
    /* clang-format on */

    ASSERT_EQ(0, a.det());
}

/**
 * Tests determinant of 3x3 integer array computation.
 */
TEST(NDArray, Det3DInt) {
    /* clang-format off */
    ndarray::NDArray<int, 2> a({3, 3}, { 1,  2,  3,
                                      0, -4,  1,
                                      2,  3,  0});
    /* clang-format on */

    ASSERT_EQ(25, a.det());
}

/**
 * Tests determinant of 4x4 integer array computation.
 */
TEST(NDArray, Det4DInt) {
    /* clang-format off */
    ndarray::NDArray<int, 2> a({4, 4}, { 1,  0,  2, -1,
                                      3,  0,  0,  5,
                                      2,  1,  4, -3,
                                      1,  0,  5,  0});
    /* clang-format on */

    ASSERT_EQ(30, a.det());
}

/**
 * Tests determinant of 4x4 float array computation.
 */
TEST(NDArray, Det4DFloat) {
    /* clang-format off */
    ndarray::NDArray<float, 2> a({4, 4}, { 1.5,  0.0,  2.2, -1.1,
                                        3.0,  0.0,  0.0,  5.5,
                                        2.1,  1.0,  4.3, -3.3,
                                        1.0,  0.0,  5.0,  0.0});
    /* clang-format on */

    ASSERT_FLOAT_EQ(45.65, a.det());
}
