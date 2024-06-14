#pragma once

#include <barrier>
#include <chrono>
#include <initializer_list>
#include <random>
#include <thread>
#include <vector>
#ifdef __linux__
# include <sched.h>
# include <unistd.h>
#endif

struct ParallelPool {
    ParallelPool(std::initializer_list<void (*)()> entries)
        : m_threads(entries.size()),
          m_barrier(entries.size()) {
        std::size_t i = 0;
        std::mt19937 rng(
            std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<int> uni(
            0, std::thread::hardware_concurrency() - 1);
        for (auto entry: entries) {
            m_threads[i] =
                std::jthread(&ParallelPool::start, this, entry, uni(rng));
            ++i;
        }
    }

    void join() {
        for (auto &&t: m_threads) {
            t.join();
        }
    }

private:
    void start(void (*entry)(), int i) {
        // barrier 是为了保证所有线程尽可能同时开始运行，不要一前一后
        // sched_setaffinity 是为了绑定线程到不同的 CPU
        // 核心上，保证线程之间是并行而不是并发
#ifdef __linux__
        cpu_set_t cpu;
        CPU_ZERO(&cpu);
        CPU_SET(i, &cpu);
        sched_setaffinity(gettid(), sizeof(cpu), &cpu); // 绑定线程到 0 号核心
#endif
        for (int t = 0; t < 10; ++t) {
            m_barrier.arrive_and_wait(); // 等待其他线程就绪
        }
        entry();
    }

    std::vector<std::jthread> m_threads;
    std::barrier<> m_barrier;
};
