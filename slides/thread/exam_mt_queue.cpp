#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
using namespace std;

template <class T>
struct mt_queue {
private:
    std::deque<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::condition_variable m_cvFull;

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
        m_cvFull.notify_one();
        return value;
    }

    // 尝试取数据，不阻塞，如果没有，返回 nullopt
    std::optional<T> try_pop() {
        std::unique_lock lock(m_mutex);
        if (m_queue.empty())
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        m_cvFull.notify_one();
        return value;
    }

    // 尝试取数据，如果没有，等待一段时间，超时返回 nullopt
    std::optional<T> try_pop_for(std::chrono::steady_clock::duration timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [&] { return !m_queue.empty(); }))
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        m_cvFull.notify_one();
        return value;
    }

    // 尝试取数据，如果没有，等待直到时间点，超时返回 nullopt
    std::optional<T> try_pop_until(std::chrono::steady_clock::time_point timeout) {
        std::unique_lock lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [&] { return !m_queue.empty(); }))
            return std::nullopt;
        T value = std::move(m_queue.back());
        m_queue.pop_back();
        m_cvFull.notify_one();
        return value;
    }
};

mt_queue<string> a;

void t1() {
    a.push("啊");
    this_thread::sleep_for(1s);
    a.push("小彭老师");
    this_thread::sleep_for(1s);
    a.push("真伟大呀");
    this_thread::sleep_for(1s);
    a.push("EXIT");
}

void t2() {
    while (1) {
        auto msg = a.pop();
        if (msg == "EXIT") break;
        cout << "t2 收到消息：" << msg << '\n';
    }
}

int main() {
    vector<jthread> pool;
    pool.push_back(jthread(t1));
    pool.push_back(jthread(t2));
    pool.clear();
    return 0;
}
