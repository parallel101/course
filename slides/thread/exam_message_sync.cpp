#include "mtqueue.hpp"
#include <iostream>
#include <optional>
#include <thread>
#include <termios.h>
#include <vector>
using namespace std;

mt_queue<std::optional<int>> counter_queue;

void counter_thread() {
    int counter = 0;
    int complete_count = 0;
    while (complete_count < 10) {
        auto i = counter_queue.pop();
        if (i == std::nullopt) {
            complete_count++;
        } else {
            counter += i.value();
        }
    }
    cout << "counter: " << counter << '\n';
}

void compute(int beg, int end) {
    for (int i = beg; i < end; ++i) {
        counter_queue.push(i);
    }
    counter_queue.push(std::nullopt);
}

int main() {
    vector<jthread> pool;
    for (int i = 0; i < 10000; i += 1000) {
        pool.push_back(jthread(compute, i, i + 1000));
    }
    pool.push_back(jthread(counter_thread));
    pool.clear();
    return 0;
}
