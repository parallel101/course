#include "mtqueue.hpp"
#include <iostream>
#include <thread>
#include <termios.h>
#include <variant>
#include <vector>
using namespace std;

mt_queue<optional<int>> counter_queue;
mt_queue<monostate> finish_queue;

void counter_thread() {
    int counter = 0;
    while (true) {
        if (auto i = counter_queue.pop()) {
            counter += i.value();
        } else {
            cout << "counter_thread 收到结束消息\n";
            break;
        }
    }
    cout << "最终 counter: " << counter << '\n';
}

void compute(int beg, int end) {
    for (int i = beg; i < end; ++i) {
        counter_queue.push(i);
    }
    cout << "compute " << beg << " ~ " << end << " 计算结束，等待大部队会师\n";
    finish_queue.push(monostate());
}

void finish_thread() {
    cout << "finish_thread 等待会师\n";
    for (int i = 0; i < 10; i++) {
        (void)finish_queue.pop();
    }
    cout << "finish_thread 会师成功，正在发送结束消息\n";
    counter_queue.push(nullopt);
}

int main() {
    vector<jthread> pool;
    for (int i = 0; i < 10000; i += 1000) {
        pool.push_back(jthread(compute, i, i + 1000));
    }
    pool.push_back(jthread(finish_thread));
    pool.push_back(jthread(counter_thread));
    pool.clear();
    return 0;
}
