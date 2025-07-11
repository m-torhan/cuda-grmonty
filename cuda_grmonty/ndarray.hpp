/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/linalg.hpp"

namespace ndarray {

/**
 * @brief N-dimensional array.
 */
template <typename T>
class NDArray {
public:
    NDArray() : data_(nullptr), shape_({}), index_({}) {}
    explicit NDArray(const std::vector<unsigned int> &shape);
    NDArray(const std::vector<unsigned int> &shape, const std::vector<T> &values);
    NDArray(const std::initializer_list<unsigned int> &shape) : NDArray(std::vector<unsigned int>(shape)) {}
    NDArray(const NDArray<T> &other);
    NDArray(NDArray<T> &&other) = default;
    ~NDArray() {}

    NDArray<T> &operator=(const NDArray<T> &other);
    NDArray<T> &operator=(NDArray<T> &&other) = default;
    T operator=(T other);

    NDArray<T> operator[](const std::vector<int> &index) const;

    /**
     * @brief Converts zero-dimensional array to single number.
     */
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
    std::vector<unsigned int> shape() const;

    /**
     * @brief Reads values from input stream.
     */
    template <typename U>
    friend std::istream &operator>>(std::istream &is, const NDArray<U> &array);

    /**
     * @brief Creates array of given shape filled with zeros.
     *
     * @param shape Shape of the created array.
     */
    static NDArray<T> zeros(const std::initializer_list<unsigned int> &shape);

    /**
     * @brief Creates array of given shape filled with ones.
     *
     * @param shape Shape of the created array.
     */
    static NDArray<T> ones(const std::initializer_list<unsigned int> &shape);

    /**
     * @brief 2D array determinant.
     */
    T det() const;

private:
    unsigned int flat_index(std::vector<int> index) const;
    unsigned int flat_index(int index) const;

    std::shared_ptr<T[]> data_;
    std::vector<unsigned int> shape_;
    std::vector<int> index_;
};

template <typename T>
NDArray<T>::NDArray(const std::vector<unsigned int> &shape) : shape_(shape) {
    unsigned int size = 1;
    for (const auto &s : shape) {
        size *= s;
    }
    data_ = std::shared_ptr<T[]>(new T[size]);
    index_.resize(shape.size(), -1);
}

template <typename T>
NDArray<T>::NDArray(const std::vector<unsigned int> &shape, const std::vector<T> &values) : shape_(shape) {
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
}

template <typename T>
NDArray<T>::NDArray(const NDArray<T> &other) : shape_(other.shape()) {
    data_ = std::shared_ptr<T[]>(new T[other.size()]);

    for (int i = 0; i < static_cast<int>(other.size()); ++i) {
        data_[i] = other.data_[other.flat_index(i)];
    }
    index_.resize(other.ndim(), -1);
}

template <typename T>
NDArray<T> &NDArray<T>::operator=(const NDArray<T> &other) {
    if (this == &other) {
        return *this;
    }
    auto this_shape = shape();
    auto other_shape = other.shape();
    if (data_ == nullptr) {
        data_ = std::shared_ptr<T[]>(new T[other.size()]);
        shape_ = other.shape();
        index_.resize(shape_.size(), -1);
    } else {
        if (this_shape.size() != other_shape.size()) {
            throw std::invalid_argument("Provided ndarrays have different number of dimensions");
        }
        if (!std::equal(this_shape.begin(), this_shape.end(), other_shape.begin())) {
            throw std::invalid_argument("Provided ndarrays have different shapes");
        }
    }
    for (int i = 0; i < static_cast<int>(size()); ++i) {
        data_[flat_index(i)] = other.data_[other.flat_index(i)];
    }

    return *this;
}

template <typename T>
T NDArray<T>::operator=(T other) {
    for (int i = 0; i < static_cast<int>(size()); ++i) {
        data_[flat_index(i)] = other;
    }
    return other;
}

template <typename T>
NDArray<T> NDArray<T>::operator[](const std::vector<int> &index) const {
    if (index.size() > ndim()) {
        throw std::invalid_argument("Specified index has too many values");
    }

    NDArray<T> ret;
    ret.data_ = data_;
    ret.shape_ = shape_;
    ret.index_.clear();
    for (int i = 0, j = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (index_[i] == -1 && j < static_cast<int>(index.size())) {
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

    return ret;
}

template <typename T>
NDArray<T>::operator T() const {
    if (0 != ndim()) {
        throw std::invalid_argument("Cannot convert ndarray with ndim larger than 0");
    }
    return data_[flat_index({})];
}

template <typename T>
unsigned int NDArray<T>::ndim() const {
    unsigned int ndim = 0;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (index_[i] == -1) {
            ndim += 1;
        }
    }
    return ndim;
}

template <typename T>
unsigned int NDArray<T>::size() const {
    unsigned int size = 1;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (index_[i] == -1) {
            size *= shape_[i];
        }
    }
    return size;
}

template <typename T>
std::vector<unsigned int> NDArray<T>::shape() const {
    std::vector<unsigned int> shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (index_[i] == -1) {
            shape.push_back(shape_[i]);
        }
    }
    return shape;
}

template <typename T>
std::istream &operator>>(std::istream &is, const NDArray<T> &array) {
    for (int i = 0; i < static_cast<int>(array.size()); ++i) {
        T value;
        is >> value;
        array.data_[array.flat_index(i)] = value;
    }
    return is;
}

template <typename T>
NDArray<T> NDArray<T>::zeros(const std::initializer_list<unsigned int> &shape) {
    NDArray<T> arr(shape);

    arr = 0;

    return arr;
}

template <typename T>
NDArray<T> NDArray<T>::ones(const std::initializer_list<unsigned int> &shape) {
    NDArray<T> arr(shape);

    arr = 1;

    return arr;
}

template <typename T>
T NDArray<T>::det() const {
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

template <typename T>
unsigned int NDArray<T>::flat_index(std::vector<int> index) const {
    if (index.size() != ndim()) {
        throw std::invalid_argument("Specified index has invalid size");
    }

    std::vector<unsigned int> final_index;
    for (int i = 0, j = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (index_[i] == -1) {
            final_index.push_back(index[j] >= 0 ? index[j] : index[j] + shape_[i]);
            ++j;
        } else {
            final_index.push_back(index_[i]);
        }
    }

    unsigned int flat_index = 0;
    unsigned int shape_prod = 1;
    for (int i = static_cast<int>(final_index.size()) - 1; i >= 0; --i) {
        flat_index += shape_prod * final_index[i];
        shape_prod *= shape_[i];
    }

    return flat_index;
}

template <typename T>
unsigned int NDArray<T>::flat_index(int index) const {
    if (index < 0) {
        index += size();
    }

    std::vector<unsigned int> array_shape = shape();
    std::vector<int> non_flat_index;

    for (int i = static_cast<int>(array_shape.size()) - 1; i >= 0; --i) {
        non_flat_index.push_back(index % array_shape[i]);
        index /= array_shape[i];
    }
    std::reverse(non_flat_index.begin(), non_flat_index.end());

    return flat_index(non_flat_index);
}

}; /* namespace ndarray */
