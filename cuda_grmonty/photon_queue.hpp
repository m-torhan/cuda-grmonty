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
    /**
     * @brief Construct a PhotonQueue with a maximum capacity.
     *
     * @param max_size Maximum number of photons that can be stored.
     */
    explicit PhotonQueue(size_t max_size) : max_size_(max_size) {}

    /**
     * @brief Deleted copy constructor (non-copyable).
     */
    PhotonQueue(const PhotonQueue &) = delete;

    /**
     * @brief Deleted copy assignment operator (non-copyable).
     */
    PhotonQueue &operator=(const PhotonQueue &) = delete;

    /**
     * @brief Enqueue a photon into the queue.
     *
     * Blocks if the queue is full until space becomes available.
     *
     * @param photon Photon to enqueue.
     */
    void enqueue(photon::Photon photon);

    /**
     * @brief Force enqueue a photon into the queue.
     *
     * Adds photon regardless of queue capacity (may overwrite or bypass limit).
     *
     * @param photon Photon to enqueue.
     */
    void force_enqueue(photon::Photon photon);

    /**
     * @brief Dequeue a photon from the queue.
     *
     * Blocks if the queue is empty until a photon becomes available.
     *
     * @return Photon removed from the queue.
     */
    photon::Photon dequeue();

    /**
     * @brief Check if the queue is empty.
     *
     * @return True if the queue has no photons, false otherwise.
     */
    bool empty() const { return q_.empty(); }

    /**
     * @brief Returns queue size.
     *
     * @return Queue size.
     */
    size_t size() const { return q_.size(); }

private:
    /**
     * @brief Maximum capacity of the queue.
     */
    size_t max_size_;

    /**
     * @brief Internal queue container storing photons.
     */
    std::queue<photon::Photon> q_;

    /**
     * @brief Mutex for thread-safe access.
     */
    mutable std::mutex m_;

    /**
     * @brief Condition variable for enqueue blocking.
     */
    std::condition_variable enqueue_cv_;

    /**
     * @brief Condition variable for dequeue blocking.
     */
    std::condition_variable dequeue_cv_;
};

}; /* namespace photon */
