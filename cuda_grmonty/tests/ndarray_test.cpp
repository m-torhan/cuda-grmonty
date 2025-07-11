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

typedef std::vector<unsigned int> Shape; /* initializer_list cannot be param of parametrized test */
typedef std::vector<int> Index;
typedef std::tuple<Shape, Index, int> ShapeIndexValue;

class NDArrayShapeFixture : public ::testing::TestWithParam<Shape> {};

class NDArrayShapeIndexValueFixture : public ::testing::TestWithParam<ShapeIndexValue> {};

static std::string gen_shape_str(const testing::TestParamInfo<NDArrayShapeFixture::ParamType> &info);

static std::string
gen_shape_index_value_str(const testing::TestParamInfo<NDArrayShapeIndexValueFixture::ParamType> &info);

/**
 * Tests constructor with various shapes.
 */
TEST_P(NDArrayShapeFixture, Construct) {
    auto shape = GetParam();

    ndarray::NDArray<int> a(shape);
}

/**
 * Tests constructor with values.
 */
TEST(NDArray, InitializeWithValues) {
    ndarray::NDArray<int> a({2, 3}, {0, 1, 2, 3, 4, 5});

    ASSERT_EQ(0, (a[{0, 0}]));
    ASSERT_EQ(1, (a[{0, 1}]));
    ASSERT_EQ(2, (a[{0, 2}]));
    ASSERT_EQ(3, (a[{1, 0}]));
    ASSERT_EQ(4, (a[{1, 1}]));
    ASSERT_EQ(5, (a[{1, 2}]));
}

/**
 * Tests constructor with values.
 */
TEST(NDArray, InitializeWithValuesInvalidShape) {
    EXPECT_THROW(({ ndarray::NDArray<int> a({2, 4}, {0, 1, 2, 3, 4, 5}); }), std::invalid_argument);
}

/**
 * Tests copying to uninitialized array.
 */
TEST(NDArray, CopyToUninitializedArray) {
    ndarray::NDArray<int> a({2, 3}, {0, 1, 2, 3, 4, 5});
    ndarray::NDArray<int> b;

    b = a;

    ASSERT_EQ(0, (b[{0, 0}]));
    ASSERT_EQ(1, (b[{0, 1}]));
    ASSERT_EQ(2, (b[{0, 2}]));
    ASSERT_EQ(3, (b[{1, 0}]));
    ASSERT_EQ(4, (b[{1, 1}]));
    ASSERT_EQ(5, (b[{1, 2}]));
}

/**
 * Tests zero dimensional array.
 */
TEST(NDArray, ZeroDim) {
    ndarray::NDArray<int> a({});

    a = 0;

    ASSERT_EQ(0, a);
}

/**
 * Tests retrieving ndim of ndarray.
 */
TEST_P(NDArrayShapeFixture, Ndim) {
    auto shape = GetParam();

    ndarray::NDArray<int> a(shape);

    ASSERT_EQ(shape.size(), a.ndim());
}

/**
 * Tests retrieving size of ndarray.
 */
TEST_P(NDArrayShapeFixture, Size) {
    auto shape = GetParam();
    unsigned int size = 1;
    for (auto &s : shape) {
        size *= s;
    }

    ndarray::NDArray<int> a(shape);

    ASSERT_EQ(size, a.size());
}

/**
 * Tests retrieving shape of ndarray.
 */
TEST_P(NDArrayShapeFixture, Shape) {
    auto shape = GetParam();

    ndarray::NDArray<int> a(shape);

    std::vector<unsigned int> shape_vec(shape);
    std::vector<unsigned int> a_shape = a.shape();

    ASSERT_EQ(shape.size(), a_shape.size());
    for (int i = 0; i < static_cast<int>(shape_vec.size()); ++i) {
        ASSERT_EQ(shape_vec[i], a_shape[i]);
    }
}

/* clang-format off */
INSTANTIATE_TEST_SUITE_P(NDArray, NDArrayShapeFixture,
                         ::testing::Values(
                            Shape{2u, 3u},
                            Shape{1u, 1u},
                            Shape{},
                            Shape{3u, 4u, 5u},
                            Shape{7u, 4u, 3u, 6u, 2u}),
                         gen_shape_str);
/* clang-format on */

/**
 * Tests setting and getting single values at index.
 */
TEST_P(NDArrayShapeIndexValueFixture, SetGetIndex) {
    ndarray::NDArray<int> a({2, 3});

    a[{0, 0}] = 1;
    ASSERT_EQ(1, (a[{0, 0}]));
}

/* clang-format off */
INSTANTIATE_TEST_SUITE_P(NDArray, NDArrayShapeIndexValueFixture,
                         ::testing::Values(
                            ShapeIndexValue{{2u, 3u}, {1, 1}, 1},
                            ShapeIndexValue{{2u, 3u}, {1, 1}, -2},
                            ShapeIndexValue{{2u, 3u}, {0, 2}, 3},
                            ShapeIndexValue{{2u, 3u}, {1, 0}, 4},
                            ShapeIndexValue{{2u, 3u}, {1, -2}, 5},
                            ShapeIndexValue{{2u, 3u}, {-1, 2}, 6},
                            ShapeIndexValue{{2u, 3u, 4u}, {1, 2, 3}, -7}),
                         gen_shape_index_value_str);
/* clang-format on */

/**
 * Tests setting single values at index with negative values.
 */
TEST(NDArray, SetIndexNegative) {
    ndarray::NDArray<int> a({2, 3});

    a[{-1, -3}] = 1;
    ASSERT_EQ(1, (a[{1, 0}]));
}

/**
 * Tests getting single values at index with negative values.
 */
TEST(NDArray, GetIndexNegative) {
    ndarray::NDArray<int> a({2, 3});

    a[{1, 0}] = 1;
    ASSERT_EQ(1, (a[{-1, -3}]));
}

/**
 * Tests setting single values at invalid index.
 */
TEST(NDArray, SetIndexInvalid) {
    ndarray::NDArray<int> a({2, 3});

    EXPECT_THROW(({ a[{0, 5}] = 1; }), std::invalid_argument);
}

/**
 * Tests getting single values at invalid index.
 */
TEST(NDArray, GetIndexInvalid) {
    ndarray::NDArray<int> a({2, 3});

    EXPECT_THROW(({ a[{0, 5}]; }), std::invalid_argument);
}

/**
 * Tests getting single values at invalid index.
 */
TEST(NDArray, GetIndexInvalidDim) {
    ndarray::NDArray<int> a({2, 3});

    EXPECT_THROW(({ a[{0, 5, 1}]; }), std::invalid_argument);
}

/**
 * Tests assignment.
 */
TEST(NDArray, Assign) {
    ndarray::NDArray<int> a({2, 3});
    ndarray::NDArray<int> b({2, 3});

    a = 0;

    b = a;

    ASSERT_EQ(0, (b[{0, 0}]));
    ASSERT_EQ(0, (b[{0, 1}]));
    ASSERT_EQ(0, (b[{0, 2}]));
    ASSERT_EQ(0, (b[{1, 0}]));
    ASSERT_EQ(0, (b[{1, 1}]));
    ASSERT_EQ(0, (b[{1, 2}]));
}

/**
 * Tests assignment with invalid ndim.
 */
TEST(NDArray, AssignInvalidNdim) {
    ndarray::NDArray<int> a({2, 3});
    ndarray::NDArray<int> b({2, 3, 1});

    a = 0;

    EXPECT_THROW({ b = a; }, std::invalid_argument);
}

/**
 * Tests assignment with invalid shape.
 */
TEST(NDArray, AssignInvalidShape) {
    ndarray::NDArray<int> a({2, 3});
    ndarray::NDArray<int> b({2, 4});

    a = 0;

    EXPECT_THROW({ b = a; }, std::invalid_argument);
}

/**
 * Tests converting 0-dim array to number.
 */
TEST(NDArray, ConvertToNumber) {
    ndarray::NDArray<int> a({});

    a = 0;

    int b = a;

    ASSERT_EQ(0, b);
}

/**
 * Tests converting array with invalid number of dimensions to number.
 */
TEST(NDArray, ConvertToNumberInvalidDim) {
    ndarray::NDArray<int> a({2, 3});

    a = 0;

    EXPECT_THROW(
        {
            int b = a;
            (void)b;
        },
        std::invalid_argument);
}

/**
 * Tests setting whole ndarray with single value.
 */
TEST(NDArray, SetWholeNDArrayWithSingleValue) {
    ndarray::NDArray<int> a({2, 3});

    a = 0;
    ASSERT_EQ(0, (a[{0, 0}]));
    ASSERT_EQ(0, (a[{0, 1}]));
    ASSERT_EQ(0, (a[{0, 2}]));
    ASSERT_EQ(0, (a[{1, 0}]));
    ASSERT_EQ(0, (a[{1, 1}]));
    ASSERT_EQ(0, (a[{1, 2}]));
}

/**
 * Tests retrieveing subarray from an array.
 */
TEST(NDArray, Subarray) {
    ndarray::NDArray<int> a({2, 3});

    a[{0, 0}] = 0;
    a[{0, 1}] = 1;
    a[{0, 2}] = 2;

    ndarray::NDArray<int> b = a[{0}];

    ASSERT_EQ(3, b.size());
    ASSERT_EQ(1, b.ndim());
    ASSERT_EQ(3, b.shape()[0]);
    ASSERT_EQ(0, b[{0}]);
    ASSERT_EQ(1, b[{1}]);
    ASSERT_EQ(2, b[{2}]);
}

/**
 * Tests retrieveing subarray copy from an array.
 */
TEST(NDArray, SubarrayCopy) {
    ndarray::NDArray<int> a({2, 3});

    a[{0, 0}] = 0;
    a[{0, 1}] = 1;
    a[{0, 2}] = 2;

    ndarray::NDArray<int> b({4, 4});
    b[{0, 0}] = -1;

    ASSERT_EQ(16, b.size());
    ASSERT_EQ(2, b.ndim());

    b = a[{0}];

    ASSERT_EQ(3, b.size());
    ASSERT_EQ(1, b.ndim());
    ASSERT_EQ(3, b.shape()[0]);
    ASSERT_EQ(0, b[{0}]);
    ASSERT_EQ(1, b[{1}]);
    ASSERT_EQ(2, b[{2}]);
}

/**
 * Tests setting subarray.
 */
TEST(NDArray, SetSubarray) {
    ndarray::NDArray<int> a({2, 3});
    ndarray::NDArray<int> b({3});

    b[{0}] = 0;
    b[{1}] = 1;
    b[{2}] = 2;

    a[{0}] = b;

    ASSERT_EQ(0, (a[{0, 0}]));
    ASSERT_EQ(1, (a[{0, 1}]));
    ASSERT_EQ(2, (a[{0, 2}]));
}

/**
 * Tests setting subarray with single value.
 */
TEST(NDArray, SetSubarrayWithSingleValue) {
    ndarray::NDArray<int> a({2, 3});

    a[{0}] = 0;

    ASSERT_EQ(0, (a[{0, 0}]));
    ASSERT_EQ(0, (a[{0, 1}]));
    ASSERT_EQ(0, (a[{0, 2}]));
}

/**
 * Tests move assignment operator.
 */
TEST(NDArray, Move) {
    ndarray::NDArray<int> a({2, 3});

    a[{0, 0}] = 0;
    a[{0, 1}] = 1;
    a[{0, 2}] = 2;
    a[{1, 0}] = 3;
    a[{1, 1}] = 4;
    a[{1, 2}] = 5;

    ndarray::NDArray<int> b({4, 4});
    b[{0, 0}] = -1;

    b = std::move(a);

    ASSERT_EQ(6, b.size());
    ASSERT_EQ(2, b.ndim());
    ASSERT_EQ(2, b.shape()[0]);
    ASSERT_EQ(3, b.shape()[1]);
    ASSERT_EQ(0, (b[{0, 0}]));
    ASSERT_EQ(1, (b[{0, 1}]));
    ASSERT_EQ(2, (b[{0, 2}]));
    ASSERT_EQ(3, (b[{1, 0}]));
    ASSERT_EQ(4, (b[{1, 1}]));
    ASSERT_EQ(5, (b[{1, 2}]));
}

/**
 * Tests reading data from stream.
 */
TEST(NDArray, ReadStream) {
    ndarray::NDArray<int> a({2, 3});

    std::stringstream s;
    s << 0 << " " << 1 << " " << 2 << " " << 3 << " " << 4 << " " << 5;

    s >> a;

    ASSERT_EQ(0, (a[{0, 0}]));
    ASSERT_EQ(1, (a[{0, 1}]));
    ASSERT_EQ(2, (a[{0, 2}]));
    ASSERT_EQ(3, (a[{1, 0}]));
    ASSERT_EQ(4, (a[{1, 1}]));
    ASSERT_EQ(5, (a[{1, 2}]));
}

/**
 * Tests creating array filled with zeros.
 */
TEST(NDArray, Zeros) {
    auto a = ndarray::NDArray<int>::zeros({2, 3});

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            int value = a[{i, j}];
            ASSERT_EQ(0, value);
        }
    }
}

/**
 * Tests creating array filled with ones.
 */
TEST(NDArray, Ones) {
    auto a = ndarray::NDArray<int>::ones({2, 3});

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            int value = a[{i, j}];
            ASSERT_EQ(1, value);
        }
    }
}

/**
 * Tests determinant of 1x1 integer array computation.
 */
TEST(NDArray, Det1DInt) {
    ndarray::NDArray<int> a({1, 1}, {6});

    ASSERT_EQ(6, a.det());
}

/**
 * Tests determinant of 2x2 integer array computation.
 */
TEST(NDArray, Det2DInt) {
    /* clang-format off */
    ndarray::NDArray<int> a({2, 2}, {3, 5, 2, 4});
    /* clang-format on */

    ASSERT_EQ(2, a.det());
}

/**
 * Tests determinant of singular 2x2 integer array computation.
 */
TEST(NDArray, Det2DIntSingular) {
    /* clang-format off */
    ndarray::NDArray<int> a({2, 2}, { 3,  5,
                                      6, 10});
    /* clang-format on */

    ASSERT_EQ(0, a.det());
}

/**
 * Tests determinant of 3x3 integer array computation.
 */
TEST(NDArray, Det3DInt) {
    /* clang-format off */
    ndarray::NDArray<int> a({3, 3}, { 1,  2,  3,
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
    ndarray::NDArray<int> a({4, 4}, { 1,  0,  2, -1,
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
    ndarray::NDArray<float> a({4, 4}, { 1.5,  0.0,  2.2, -1.1,
                                        3.0,  0.0,  0.0,  5.5,
                                        2.1,  1.0,  4.3, -3.3,
                                        1.0,  0.0,  5.0,  0.0});
    /* clang-format on */

    ASSERT_FLOAT_EQ(45.65, a.det());
}

static std::string gen_shape_str(const testing::TestParamInfo<NDArrayShapeFixture::ParamType> &info) {
    const Shape &shape = info.param;

    std::stringstream name;
    if (shape.size() == 0) {
        return std::string("empty_shape");
    }
    std::copy(shape.begin(), shape.end(), std::ostream_iterator<unsigned int>(name, "_"));
    name << "shape";
    std::string name_str = name.str();
    std::replace(name_str.begin(), name_str.end(), '-', 'm');
    return name_str;
}

static std::string
gen_shape_index_value_str(const testing::TestParamInfo<NDArrayShapeIndexValueFixture::ParamType> &info) {
    const Shape &shape = std::get<0>(info.param);
    const Index &index = std::get<1>(info.param);
    const int &value = std::get<2>(info.param);

    std::stringstream name;
    if (shape.size() == 0) {
        name << "empty_shape";
    } else {
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<unsigned int>(name, "_"));
        name << "shape_";
    }
    std::copy(index.begin(), index.end(), std::ostream_iterator<int>(name, "_"));
    name << "index_";
    name << value;
    name << "_value";
    std::string name_str = name.str();
    std::replace(name_str.begin(), name_str.end(), '-', 'm');
    return name_str;
}
