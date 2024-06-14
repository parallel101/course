#pragma once

// 小彭老师自主研发的一款多线程安全验证器

#include <barrier>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

template <class State>
void mtverify(std::initializer_list<void (State::*)()> tasks, size_t epochs = 10, size_t perround = 10000) {
    using Repr = decltype(std::declval<State &>().repr());
    State state;
    std::barrier<> sync(tasks.size() + 1);
    std::map<Repr, int> counts;

    std::vector<std::jthread> threads(tasks.size());
    std::cout << "[共 " << epochs * perround << " 次]\n";

    for (size_t t = 0; t < epochs; t++) {
        auto it = tasks.begin();
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i] = std::jthread([task = *it++, perround, &sync, &state] {
                for (size_t i = 0; i < 4; i++) {
                    sync.arrive_and_wait();
                }
                for (size_t i = 0; i < perround; i++) {
                    sync.arrive_and_wait();
                    (state.*task)();
                    sync.arrive_and_wait();
                }
            });
        }

        for (size_t i = 0; i < 4; i++) {
            sync.arrive_and_wait();
        }
        for (size_t i = 0; i < perround; i++) {
            state.~State();
            new (&state) State();
            sync.arrive_and_wait();
            sync.arrive_and_wait();
            ++counts[state.repr()];
        }
        for (auto &&thread: threads) {
            thread.join();
        }
    }

    for (auto &&[k, v]: counts) {
        std::cout << "最终结果为 " << k << " 出现了 " << v << " 次\n";
    }
    for (auto &&[k, v]: counts) {
        std::cout << "[" << k << "] ";
        double percent = (double)v / (epochs * perround);
        int nbars = (int)(percent * 50 + 0.5);
        for (int i = 0; i < nbars; i++) {
            std::cout << '=';
        }
        std::cout << " " << (double)(int)(percent * 1000 + 0.5) / 10
            << "%\n";
    }
}
