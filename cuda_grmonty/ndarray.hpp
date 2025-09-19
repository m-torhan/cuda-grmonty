/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "cuda_grmonty/linalg.hpp"

namespace ndarray {

/**
 * @brief N-dimensional array container with basic indexing and assignment operators.
 *
 * Provides multidimensional storage using a flat vector underneath. Supports copy, move, and element-wise access
 * through operator() overloads. Dimension checks are enforced via template metaprogramming.
 *
 * @tparam T Element type stored in the array.
 * @tparam N Number of dimensions.
 */
template <typename T, unsigned int N>
class NDArray {
public:
    /**
     * @brief Type alias for the index representation of an N-dimensional array.
     *
     * Each element of the array represents the size in that dimension.
     */
    using Index = std::array<int, N>;

    /**
     * @brief Default constructor, creates an uninitialized array (no allocated storage).
     */
    NDArray() : data_(nullptr) {}

    /**
     * @brief Construct an array with the given shape.
     *
     * @param shape Sizes of each dimension.
     */
    explicit NDArray(const Index &shape);

    /**
     * @brief Construct an array with a given shape and initial values.
     *
     * @param shape  Sizes of each dimension.
     * @param values Initial values to populate the array with.
     */
    NDArray(const Index &shape, const std::vector<T> &values);

    /**
     * @brief Copy constructor.
     */
    NDArray(const NDArray<T, N> &other);

    /**
     * @brief Prevent construction from NDArray of different dimension.
     *
     * @tparam M Other array dimensionality (must differ from N).
     */
    template <unsigned int M, typename = std::enable_if_t<N != M>>
    NDArray(const NDArray<T, M> &other) = delete;

    /**
     * @brief Move constructor.
     */
    NDArray(NDArray<T, N> &&other) = default;

    /**
     * @brief Destructor.
     */
    ~NDArray() {}

    /**
     * @brief Copy assignment operator.
     */
    NDArray<T, N> &operator=(const NDArray<T, N> &other);

    /**
     * @brief Prevent assignment from NDArray of different dimension.
     *
     * @tparam M Other array dimensionality (must differ from N).
     */
    template <unsigned int M, typename = std::enable_if_t<N != M>>
    NDArray<T, N> &operator=(NDArray<T, M> &other) = delete;

    /**
     * @brief Move assignment operator.
     */
    NDArray<T, N> &operator=(NDArray<T, N> &&other) = default;

    /**
     * @brief Assign scalar value to all elements of the array.
     */
    T operator=(T other);

    /**
     * @brief Return a lower-dimensional NDArray view (non-const).
     *
     * Allows partial indexing with fewer than N indices, returning an NDArray of dimension N - sizeof...(Indices).
     *
     * @tparam Indices Index types (variadic).
     * @param indices  Partial indices specifying the slice.
     *
     * @return Sub-array view of lower dimensionality.
     */
    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) < N>>
    NDArray<T, N - sizeof...(Indices)> operator()(Indices... indices) noexcept;

    /**
     * @brief Return a lower-dimensional NDArray view (const version).
     *
     * Allows partial indexing with fewer than N indices, returning an NDArray of dimension N - sizeof...(Indices).
     *
     * @tparam Indices Index types (variadic).
     * @param indices  Partial indices specifying the slice.
     *
     * @return Const sub-array view of lower dimensionality.
     */
    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) < N>>
    const NDArray<T, N - sizeof...(Indices)> operator()(Indices... indices) const noexcept;

    /**
     * @brief Return an element of the array (const version).
     *
     * Requires exactly N indices to fully specify the element position.
     *
     * @tparam Indices Index types (variadic).
     * @param indices  Indices for each dimension.
     *
     * @return Element value at the given indices.
     */
    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == N>>
    T operator()(Indices... indices) const noexcept;

    /**
     * @brief Return a reference to an element of the array (non-const).
     *
     * Requires exactly N indices to fully specify the element position.
     *
     * @tparam Indices Index types (variadic).
     * @param indices  Indices for each dimension.
     *
     * @return Reference to the element at the given indices.
     */
    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == N>>
    T &operator()(Indices... indices) noexcept;

    /**
     * @brief Convert a zero-dimensional array into a scalar value.
     *
     * This operator allows an NDArray<...,0> to be implicitly converted into its contained value.
     *
     * @tparam K  Dimension parameter, must equal 0 to enable this operator.
     *
     * @return Scalar value stored in the array.
     */
    template <unsigned int K = N, typename = std::enable_if_t<K == 0>>
    operator T() const;

    /**
     * @brief Returns pointer to the underlying raw data buffer.
     *
     * @return Raw pointer to the array data.
     */
    T *data() const noexcept { return data_.get(); }

    /**
     * @brief Returns the number of dimensions of the array.
     *
     * @return Dimensionality (N).
     */
    unsigned int ndim() const noexcept { return N; }

    /**
     * @brief Returns the total number of elements in the array.
     *
     * @return Number of elements in the array.
     */
    unsigned int size() const noexcept;

    /**
     * @brief Returns the shape of the array.
     *
     * @return Array of length N representing the size of each dimension.
     */
    std::array<int, N> shape() const noexcept { return shape_; }

    /**
     * @brief Read array values from an input stream.
     *
     * Reads elements in row-major order. Shape must already be defined.
     *
     * @tparam U    Element type of the array.
     * @tparam M    Dimensionality of the array.
     * @param is    Input stream.
     * @param array Target array to populate.
     *
     * @return Reference to the input stream.
     */
    template <typename U, unsigned int M>
    friend std::istream &operator>>(std::istream &is, const NDArray<U, M> &array);

    /**
     * @brief Compute determinant of a 2D array.
     *
     * @tparam K Dimension parameter, must equal 2 to enable this method.
     *
     * @return Determinant of the 2D array.
     */
    template <unsigned int K = N, typename = std::enable_if_t<K == 2>>
    T det() const;

private:
    /**
     * @brief Compute strides for indexing based on current shape.
     *
     * Strides define the number of elements to skip in the flat buffer when advancing along each dimension.
     */
    void compute_strides();

    /**
     * @brief Compute the flat buffer index for a given N-dimensional index.
     *
     * @param index N-dimensional index in row-major order.
     *
     * @return Flat buffer offset corresponding to the index.
     */
    unsigned int flat_index(const std::array<int, N> &index) const noexcept;

    /**
     * @brief Shared pointer to array data.
     */
    std::shared_ptr<T[]> data_;

    /**
     * @brief Offset applied to the data pointer (used for views).
     */
    int data_offset_;

    /**
     * @brief Shape of the array (size of each dimension).
     */
    std::array<int, N> shape_;

    /**
     * @brief Strides for row-major indexing.
     */
    std::array<int, N> strides_;

    /**
     * @brief Allow all NDArray<T, M> specializations to access private members.
     */
    template <typename, unsigned int>
    friend class NDArray;
};

template <typename T, unsigned int N>
NDArray<T, N>::NDArray(const std::array<int, N> &shape) : shape_(shape) {
    unsigned int size = 1;
    for (const auto &s : shape) {
        size *= s;
    }
    data_ = std::shared_ptr<T[]>(new T[size]);
    data_offset_ = 0;
    compute_strides();
}

template <typename T, unsigned int N>
NDArray<T, N>::NDArray(const std::array<int, N> &shape, const std::vector<T> &values) : shape_(shape) {
    unsigned int size = 1;
    for (const auto &s : shape) {
        size *= s;
    }
    data_ = std::shared_ptr<T[]>(new T[size]);
    data_offset_ = 0;
    if (size != values.size()) {
        throw std::invalid_argument("Invalid number of values");
    }
    for (int i = 0; i < static_cast<int>(size); ++i) {
        data_[data_offset_ + i] = values[i];
    }
    compute_strides();
}

template <typename T, unsigned int N>
NDArray<T, N>::NDArray(const NDArray<T, N> &other)
    : data_(other.data_), data_offset_(other.data_offset_), shape_(other.shape()), strides_(other.strides_) {}

template <typename T, unsigned int N>
NDArray<T, N> &NDArray<T, N>::operator=(const NDArray<T, N> &other) {
    if (this == &other) {
        return *this;
    }

    std::array<int, N> this_shape = shape();
    std::array<int, N> other_shape = other.shape();

    if (data_ == nullptr) {
        data_ = other.data_;
        data_offset_ = other.data_offset_;
        shape_ = other.shape_;
        strides_ = other.strides_;
    } else {
        if (!std::equal(this_shape.begin(), this_shape.end(), other_shape.begin())) {
            throw std::invalid_argument("Provided ndarrays have different shapes");
        }
        for (int i = 0; i < static_cast<int>(size()); ++i) {
            data_[data_offset_ + i] = other.data_[other.data_offset_ + i];
        }
    }

    return *this;
}

template <typename T, unsigned int N>
T NDArray<T, N>::operator=(T other) {
    for (int i = 0; i < static_cast<int>(size()); ++i) {
        data_[data_offset_ + i] = other;
    }
    return other;
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
const NDArray<T, N - sizeof...(Indices)> NDArray<T, N>::operator()(Indices... indices) const noexcept {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};
    NDArray<T, N - sizeof...(Indices)> ret;

    ret.data_ = data_;
    ret.data_offset_ = data_offset_;

    for (int i = 0; i < static_cast<int>(index.size()); ++i) {
        ret.data_offset_ += strides_[i] * index[i];
    }
    for (size_t i = 0; i < N - index.size(); ++i) {
        ret.shape_[i] = shape_[index.size() + i];
    }

    ret.compute_strides();

    return ret;
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
NDArray<T, N - sizeof...(Indices)> NDArray<T, N>::operator()(Indices... indices) noexcept {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};
    NDArray<T, N - sizeof...(Indices)> ret;

    ret.data_ = data_;
    ret.data_offset_ = data_offset_;

    for (int i = 0; i < static_cast<int>(index.size()); ++i) {
        ret.data_offset_ += strides_[i] * index[i];
    }
    for (int i = 0; i < static_cast<int>(N - index.size()); ++i) {
        ret.shape_[i] = shape_[index.size() + i];
    }

    ret.compute_strides();

    return ret;
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
T NDArray<T, N>::operator()(Indices... indices) const noexcept {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};

    return data_[data_offset_ + flat_index(index)];
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
T &NDArray<T, N>::operator()(Indices... indices) noexcept {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};

    return data_[data_offset_ + flat_index(index)];
}

template <typename T, unsigned int N>
template <unsigned int, typename>
NDArray<T, N>::operator T() const {
    if (0 != ndim()) {
        throw std::invalid_argument("Cannot convert ndarray with ndim larger than 0");
    }
    return data_[data_offset_ + flat_index({})];
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::size() const noexcept {
    unsigned int size = 1;
    for (int i = 0; i < static_cast<int>(N); ++i) {
        size *= shape_[i];
    }
    return size;
}

template <typename T, unsigned int N>
std::istream &operator>>(std::istream &is, const NDArray<T, N> &array) {
    for (int i = 0; i < static_cast<int>(array.size()); ++i) {
        T value;
        is >> value;
        array.data_[array.data_offset_ + i] = value;
    }
    return is;
}

template <typename T, unsigned int N>
template <unsigned int, typename>
T NDArray<T, N>::det() const {
    if (ndim() != 2) {
        throw std::invalid_argument("Array should be 2-dimensional");
    }
    auto shape = this->shape();
    if (shape[0] != shape[1]) {
        throw std::invalid_argument("Array should be square");
    }
    int n = static_cast<int>(shape[0]);
    T *data = new T[n * n];

    for (int i = 0; i < static_cast<int>(size()); ++i) {
        data[i] = data_[data_offset_ + i];
    }

    T ret = linalg::matrix::det(n, data);

    delete[] data;

    return ret;
}

template <typename T, unsigned int N>
void NDArray<T, N>::compute_strides() {
    unsigned int stride = 1;
    for (int i = N - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::flat_index(const std::array<int, N> &index) const noexcept {
    unsigned int flat_index = 0;
    for (int i = 0; i < static_cast<int>(N); ++i) {
        int idx = index[i];
        flat_index += static_cast<unsigned int>(idx) * strides_[i];
    }
    return flat_index;
}

}; /* namespace ndarray */
