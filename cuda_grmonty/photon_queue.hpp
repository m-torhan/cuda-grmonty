/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include "cuda_grmonty/photon.hpp"

namespace photon {

class PhotonQueue {
public:
    explicit PhotonQueue(size_t max_size) : max_size_(max_size) {}
    PhotonQueue(const PhotonQueue &) = delete;
    PhotonQueue &operator=(const PhotonQueue &) = delete;

    void enqueue(photon::Photon photon);

    void force_enqueue(photon::Photon photon);

    photon::Photon dequeue();

    bool empty() const;

private:
    size_t max_size_;
    std::queue<photon::Photon> q_;
    mutable std::mutex m_;
    std::condition_variable enqueue_cv_;
    std::condition_variable dequeue_cv_;
};

}; /* namespace photon */
