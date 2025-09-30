/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace utils {

template <typename T>
class ConcurrentQueue {
public:
    /**
     * @brief Construct a ConcurrentQueue with a maximum capacity.
     *
     * @param max_size Maximum number of objects that can be stored.
     */
    explicit ConcurrentQueue(size_t max_size) : max_size_(max_size) {}

    /**
     * @brief Deleted copy constructor (non-copyable).
     */
    ConcurrentQueue(const ConcurrentQueue &) = delete;

    /**
     * @brief Deleted copy assignment operator (non-copyable).
     */
    ConcurrentQueue &operator=(const ConcurrentQueue &) = delete;

    /**
     * @brief Enqueue a object into the queue.
     *
     * Blocks if the queue is full until space becomes available.
     *
     * @param object Object to enqueue.
     */
    void enqueue(T object) {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.size() >= max_size_) {
            dequeue_cv_.wait(lock);
        }

        q_.push(object);
        enqueue_cv_.notify_one();
    }

    /**
     * @brief Force enqueue a object into the queue.
     *
     * Adds object regardless of queue capacity (may overwrite or bypass limit).
     *
     * @param object Object to enqueue.
     */
    void force_enqueue(T object) {
        std::lock_guard<std::mutex> lock(m_);
        q_.push(object);
        enqueue_cv_.notify_one();
    }

    /**
     * @brief Dequeue a object from the queue.
     *
     * Blocks if the queue is empty until a object becomes available.
     *
     * @return Object removed from the queue.
     */
    T dequeue() {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.empty()) {
            enqueue_cv_.wait(lock);
        }

        T object = q_.front();
        q_.pop();
        dequeue_cv_.notify_one();
        return object;
    }

    /**
     * @brief Check if the queue is empty.
     *
     * @return True if the queue has no objects, false otherwise.
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
     * @brief Internal queue container storing objects.
     */
    std::queue<T> q_;

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

}; /* namespace utils */
