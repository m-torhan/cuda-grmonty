/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
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

    NDArray() : data_(nullptr), shape_({}), index_({}) {}
    explicit NDArray(const Index &shape);
    NDArray(const Index &shape, const std::vector<T> &values);
    NDArray(const NDArray<T, N> &other);
    template <unsigned int M, typename = std::enable_if_t<N != M>>
    NDArray(const NDArray<T, N> &other) = delete;
    NDArray(NDArray<T, N> &&other) = default;
    ~NDArray() {}

    NDArray<T, N> &operator=(const NDArray<T, N> &other);
    template <unsigned int M, typename = std::enable_if_t<N != M>>
    NDArray<T, N> &operator=(NDArray<T, M> &other) = delete;
    NDArray<T, N> &operator=(NDArray<T, N> &&other) = default;
    T operator=(T other);

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) < N>>
    NDArray<T, N - sizeof...(Indices)> operator()(Indices... indices);

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) < N>>
    const NDArray<T, N - sizeof...(Indices)> operator()(Indices... indices) const;

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == N>>
    T operator()(Indices... indices) const noexcept;

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == N>>
    T &operator()(Indices... indices) noexcept;

    /**
     * @brief Converts zero-dimensional array to single number.
     */
    template <unsigned int K = N, typename = std::enable_if_t<K == 0>>
    operator T() const;

    /**
     * @brief Returns number of dimensions.
     */
    unsigned int ndim() const;

    /**
     * @brief Returns number of elements.
     */
    unsigned int size() const;

    /**
     * @brief Returns shape of the array.
     */
    std::array<int, N> shape() const;

    /**
     * @brief Converts 0-dim array to value.
     */
    T value() const;

    /**
     * @brief Reads values from input stream.
     */
    template <typename U, unsigned int M>
    friend std::istream &operator>>(std::istream &is, const NDArray<U, M> &array);

    /**
     * @brief 2D array determinant.
     */
    T det() const;

private:
    void compute_strides();
    unsigned int flat_index(int index) const noexcept;
    unsigned int flat_index(const std::array<int, N> &index) const noexcept;
    unsigned int flat_index_general(const std::array<int, N> &index) const noexcept;
    unsigned int flat_index_contiguous(const std::array<int, N> &index) const noexcept;

    std::shared_ptr<T[]> data_;
    std::vector<int> shape_;
    std::vector<int> index_;
    std::vector<int> strides_;
    bool is_contiguous_;

    /* Make all NDArray<T, M> friends (M can be any dimension) */
    template <typename, unsigned int>
    friend class NDArray;
};

template <typename T, unsigned int N>
NDArray<T, N>::NDArray(const std::array<int, N> &shape) : shape_(shape.begin(), shape.end()) {
    unsigned int size = 1;
    for (const auto &s : shape) {
        size *= s;
    }
    data_ = std::shared_ptr<T[]>(new T[size]);
    index_.resize(shape.size(), -1);
    is_contiguous_ = true;
    compute_strides();
}

template <typename T, unsigned int N>
NDArray<T, N>::NDArray(const std::array<int, N> &shape, const std::vector<T> &values)
    : shape_(shape.begin(), shape.end()) {
    unsigned int size = 1;
    for (const auto &s : shape) {
        size *= s;
    }
    data_ = std::shared_ptr<T[]>(new T[size]);
    index_.resize(shape.size(), -1);
    if (size != values.size()) {
        throw std::invalid_argument("Invalid number of values");
    }
    for (int i = 0; i < static_cast<int>(size); ++i) {
        data_[i] = values[i];
    }
    is_contiguous_ = true;
    compute_strides();
}

template <typename T, unsigned int N>
NDArray<T, N>::NDArray(const NDArray<T, N> &other) : shape_(other.shape()) {
    data_ = std::shared_ptr<T[]>(new T[other.size()]);

    for (int i = 0; i < static_cast<int>(other.size()); ++i) {
        data_[i] = other.data_[other.flat_index(i)];
    }
    index_.resize(other.ndim(), -1);
    is_contiguous_ = true;
    compute_strides();
}

template <typename T, unsigned int N>
NDArray<T, N> &NDArray<T, N>::operator=(const NDArray<T, N> &other) {
    if (this == &other) {
        return *this;
    }

    std::array<int, N> this_shape = shape();
    std::array<int, N> other_shape = other.shape();

    if (data_ == nullptr) {
        data_ = std::shared_ptr<T[]>(new T[other.size()]);
        shape_.resize(other_shape.size());
        std::copy(std::begin(other_shape), std::end(other_shape), std::begin(shape_));
        index_.resize(shape_.size(), -1);
        is_contiguous_ = true;
        compute_strides();
    } else {
        if (!std::equal(this_shape.begin(), this_shape.end(), other_shape.begin())) {
            throw std::invalid_argument("Provided ndarrays have different shapes");
        }
        is_contiguous_ = other.is_contiguous_;
    }
    for (int i = 0; i < static_cast<int>(size()); ++i) {
        data_[flat_index(i)] = other.data_[other.flat_index(i)];
    }
    compute_strides();

    return *this;
}

template <typename T, unsigned int N>
T NDArray<T, N>::operator=(T other) {
    for (int i = 0; i < static_cast<int>(size()); ++i) {
        data_[flat_index(i)] = other;
    }
    return other;
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
const NDArray<T, N - sizeof...(Indices)> NDArray<T, N>::operator()(Indices... indices) const {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};
    NDArray<T, N - sizeof...(Indices)> ret;

    ret.data_ = data_;
    ret.shape_ = shape_;
    ret.index_.clear();
    for (size_t i = 0, j = 0; i < shape_.size(); ++i) {
        if (index_[i] == -1 && j < sizeof...(Indices)) {
            int idx = index[j] >= 0 ? index[j] : shape_[i] + index[j];
            if (idx > static_cast<int>(shape_[i]) || idx < 0) {
                throw std::invalid_argument("Invalid index");
            }
            ret.index_.push_back(idx);
            ++j;
        } else {
            ret.index_.push_back(index_[i]);
        }
    }
    ret.is_contiguous_ = false;
    ret.compute_strides();

    return ret;
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
NDArray<T, N - sizeof...(Indices)> NDArray<T, N>::operator()(Indices... indices) {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};
    NDArray<T, N - sizeof...(Indices)> ret;

    ret.data_ = data_;
    ret.shape_ = shape_;
    ret.index_.clear();
    for (size_t i = 0, j = 0; i < shape_.size(); ++i) {
        if (index_[i] == -1 && j < sizeof...(Indices)) {
            int idx = index[j] >= 0 ? index[j] : shape_[i] + index[j];
            if (idx > static_cast<int>(shape_[i]) || idx < 0) {
                throw std::invalid_argument("Invalid index");
            }
            ret.index_.push_back(idx);
            ++j;
        } else {
            ret.index_.push_back(index_[i]);
        }
    }
    ret.is_contiguous_ = false;
    ret.compute_strides();

    return ret;
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
T NDArray<T, N>::operator()(Indices... indices) const noexcept {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};

    return data_[flat_index(index)];
}

template <typename T, unsigned int N>
template <typename... Indices, typename>
T &NDArray<T, N>::operator()(Indices... indices) noexcept {
    std::array<int, sizeof...(Indices)> index = {static_cast<int>(indices)...};

    return data_[flat_index(index)];
}

template <typename T, unsigned int N>
template <unsigned int, typename>
NDArray<T, N>::operator T() const {
    if (0 != ndim()) {
        throw std::invalid_argument("Cannot convert ndarray with ndim larger than 0");
    }
    return data_[flat_index({})];
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::ndim() const {
    return N;
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::size() const {
    unsigned int size = 1;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (index_[i] == -1) {
            size *= shape_[i];
        }
    }
    return size;
}

template <typename T, unsigned int N>
std::array<int, N> NDArray<T, N>::shape() const {
    std::array<int, N> shape{};
    for (int i = 0, j = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (index_[i] == -1) {
            shape[j] = shape_[i];
            ++j;
        }
    }
    return shape;
}

template <typename T, unsigned int N>
T NDArray<T, N>::value() const {
    return static_cast<T>(*this);
}

template <typename T, unsigned int N>
std::istream &operator>>(std::istream &is, const NDArray<T, N> &array) {
    for (int i = 0; i < static_cast<int>(array.size()); ++i) {
        T value;
        is >> value;
        array.data_[array.flat_index(i)] = value;
    }
    return is;
}

template <typename T, unsigned int N>
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
        data[i] = data_[flat_index(i)];
    }

    T ret = linalg::matrix::det(n, data);

    delete[] data;

    return ret;
}

template <typename T, unsigned int N>
void NDArray<T, N>::compute_strides() {
    strides_.resize(shape_.size());
    unsigned int stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::flat_index(int index) const noexcept {
    int s = size();
    if (index < 0) {
        index += s;
    }
    if (is_contiguous_) {
        return index;
    }

    std::array<int, N> array_shape = shape();
    std::array<int, N> non_flat_index;

    for (int i = N - 1; i >= 0; --i) {
        non_flat_index[i] = index % array_shape[i];
        index /= array_shape[i];
    }

    return flat_index(non_flat_index);
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::flat_index(const std::array<int, N> &index) const noexcept {
    std::array<int, N> s = shape();
    std::array<int, N> final_index;
    for (int i = 0; i < static_cast<int>(N); ++i) {
        final_index[i] = index[i] >= 0 ? index[i] : index[i] + s[i];
    }
    return is_contiguous_ ? flat_index_contiguous(index) : flat_index_general(index);
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::flat_index_general(const std::array<int, N> &index) const noexcept {
    unsigned int flat_index = 0;
    size_t j = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        unsigned int idx;
        if (index_[i] == -1) {
            int raw = index[j++];
            if (raw < 0) {
                raw += shape_[i];
            }
            idx = static_cast<unsigned int>(raw);
        } else {
            idx = static_cast<unsigned int>(index_[i]);
        }
        flat_index += idx * strides_[i];
    }
    return flat_index;
}

template <typename T, unsigned int N>
unsigned int NDArray<T, N>::flat_index_contiguous(const std::array<int, N> &index) const noexcept {
    unsigned int flat_index = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        int idx = index[i];
        if (idx < 0)
            idx += shape_[i];
        flat_index += static_cast<unsigned int>(idx) * strides_[i];
    }
    return flat_index;
}

}; /* namespace ndarray */
