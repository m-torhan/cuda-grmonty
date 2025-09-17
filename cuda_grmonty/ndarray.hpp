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
 * @brief N-dimensional array.
 */
template <typename T, unsigned int N>
class NDArray {
public:
    using Index = std::array<int, N>;

    NDArray() : data_(nullptr) {}
    explicit NDArray(const Index &shape);
    NDArray(const Index &shape, const std::vector<T> &values);
    NDArray(const NDArray<T, N> &other);
    template <unsigned int M, typename = std::enable_if_t<N != M>>
    NDArray(const NDArray<T, M> &other) = delete;
    NDArray(NDArray<T, N> &&other) = default;
    ~NDArray() {}

    NDArray<T, N> &operator=(const NDArray<T, N> &other);
    template <unsigned int M, typename = std::enable_if_t<N != M>>
    NDArray<T, N> &operator=(NDArray<T, M> &other) = delete;
    NDArray<T, N> &operator=(NDArray<T, N> &&other) = default;
    T operator=(T other);

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) < N>>
    NDArray<T, N - sizeof...(Indices)> operator()(Indices... indices) noexcept;

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) < N>>
    const NDArray<T, N - sizeof...(Indices)> operator()(Indices... indices) const noexcept;

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == N>>
    T operator()(Indices... indices) const noexcept;

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == N>>
    T &operator()(Indices... indices) noexcept;

    /**
     * @brief Converts zero-dimensional array to single number.
     */
    template <unsigned int K = N, typename = std::enable_if_t<K == 0>>
    operator T() const;

    T *data() const noexcept { return data_.get(); }

    /**
     * @brief Returns number of dimensions.
     */
    unsigned int ndim() const noexcept { return N; }

    /**
     * @brief Returns number of elements.
     */
    unsigned int size() const noexcept;

    /**
     * @brief Returns shape of the array.
     */
    std::array<int, N> shape() const noexcept { return shape_; }

    /**
     * @brief Reads values from input stream.
     */
    template <typename U, unsigned int M>
    friend std::istream &operator>>(std::istream &is, const NDArray<U, M> &array);

    /**
     * @brief 2D array determinant.
     */
    template <unsigned int K = N, typename = std::enable_if_t<K == 2>>
    T det() const;

private:
    void compute_strides();
    unsigned int flat_index(const std::array<int, N> &index) const noexcept;

    std::shared_ptr<T[]> data_;
    int data_offset_;
    std::array<int, N> shape_;
    std::array<int, N> strides_;

    /* Make all NDArray<T, M> friends (M can be any dimension) */
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
