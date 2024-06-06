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
    std::condition_variable m_cv_empty;
    std::condition_variable m_cv_full;
    std::size_t m_limit;

public:
    // 默认 (size_t)-1 表示容纳无限量元素
    mt_queue() : m_limit(static_cast<std::size_t>(-1)) {}

    // 指定最大允许堆积的元素数量，超过该数量后会阻塞
    explicit mt_queue(std::size_t limit) : m_limit(limit) {}

    // 推入数据，如果满，则阻塞
    void push(T value) {
        std::unique_lock lock(m_mutex);
        while (m_queue.size() >= m_limit)
            m_cv_full.wait(lock);
        m_queue.push_front(std::move(value));
        m_cv_empty.notify_one();
    }

    // 尝试推数据，不阻塞，如果满，返回 false
    bool try_push(T value) {
        std::unique_lock lock(m_mutex);
        if (m_queue.size() >= m_limit)
            return false;
        m_queue.push_front(std::move(value));
        m_cv_empty.notify_one();
        return true;
    }

    // 尝试推数据，如果满，等待一段时间，超时返回 false
    bool try_push_for(T value, std::chrono::steady_clock::duration timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv_full.wait_for(lock, timeout, [&] { return m_queue.size() < m_limit; }))
            return false;
        m_queue.push_front(std::move(value));
        m_cv_empty.notify_one();
        return true;
    }

    // 尝试推数据，如果满，等待直到时间点，超时返回 false
    bool try_push_until(T value, std::chrono::steady_clock::time_point timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv_full.wait_until(lock, timeout, [&] { return m_queue.size() < m_limit; }))
            return false;
        m_queue.push_front(std::move(value));
        m_cv_empty.notify_one();
        return true;
    }

    // 陷入阻塞，直到有数据
    T pop() {
        std::unique_lock lock(m_mutex);
        while (m_queue.empty())
            m_cv_empty.wait(lock);
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        m_cv_full.notify_one();
        return value;
    }

    // 尝试取数据，不阻塞，如果没有，返回 nullopt
    std::optional<T> try_pop() {
        std::unique_lock lock(m_mutex);
        if (m_queue.empty())
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        m_cv_full.notify_one();
        return value;
    }

    // 尝试取数据，如果没有，等待一段时间，超时返回 nullopt
    std::optional<T> try_pop_for(std::chrono::steady_clock::duration timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv_empty.wait_for(lock, timeout, [&] { return !m_queue.empty(); }))
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        m_cv_full.notify_one();
        return value;
    }

    // 尝试取数据，如果没有，等待直到时间点，超时返回 nullopt
    std::optional<T> try_pop_until(std::chrono::steady_clock::time_point timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv_empty.wait_for(lock, timeout, [&] { return !m_queue.empty(); }))
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        m_cv_full.notify_one();
        return value;
    }
};
