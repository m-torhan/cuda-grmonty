/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>
#include <queue>

#include "cuda_grmonty/photon.hpp"
#include "cuda_grmonty/photon_queue.hpp"

namespace photon {

void PhotonQueue::enqueue(photon::Photon photon) {
    std::unique_lock<std::mutex> lock(m_);
    while (q_.size() >= max_size_) {
        dequeue_cv_.wait(lock);
    }

    q_.push(photon);
    enqueue_cv_.notify_one();
}

void PhotonQueue::force_enqueue(photon::Photon photon) {
    std::lock_guard<std::mutex> lock(m_);
    q_.push(photon);
    enqueue_cv_.notify_one();
}

photon::Photon PhotonQueue::dequeue() {
    std::unique_lock<std::mutex> lock(m_);
    while (q_.empty()) {
        enqueue_cv_.wait(lock);
    }

    photon::Photon photon = q_.front();
    q_.pop();
    dequeue_cv_.notify_one();
    return photon;
}

bool PhotonQueue::empty() const {
    std::lock_guard<std::mutex> lock(m_);
    return q_.empty();
}

}; /* namespace photon */
