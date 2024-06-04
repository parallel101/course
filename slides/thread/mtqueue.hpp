#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

template <class T, class Deque = std::deque<T>>
struct mt_queue {
private:
    Deque m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;

public:
    // 推入数据，并通知
    void push(T value) {
        std::unique_lock lock(m_mutex);
        m_queue.push_front(std::move(value));
        m_cv.notify_one();
    }

    // 陷入阻塞，直到有数据
    T pop() {
        std::unique_lock lock(m_mutex);
        while (m_queue.empty())
            m_cv.wait(lock);
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        return value;
    }

    // 尝试取数据，不阻塞，如果没有，返回 nullopt
    std::optional<T> try_pop() {
        std::unique_lock lock(m_mutex);
        if (m_queue.empty())
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        return value;
    }

    // 尝试取数据，如果没有，等待一段时间，超时返回 nullopt
    std::optional<T> try_pop_for(std::chrono::steady_clock::duration timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [&] { return !m_queue.empty(); }))
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        return value;
    }

    // 尝试取数据，如果没有，等待直到时间点，超时返回 nullopt
    std::optional<T> try_pop_until(std::chrono::steady_clock::time_point timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [&] { return !m_queue.empty(); }))
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        return value;
    }
};
