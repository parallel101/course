#include "mtqueue.hpp"
#include <functional>
#include <iostream>
#include <thread>
#include <termios.h>
#include <vector>
#include <sstream>

using namespace std;

struct CounterState {
    int counter = 0;
    string msg = "";
    bool finish = false;
};

void counter_thread(mt_queue<function<void(CounterState &)>> &counter_queue, mt_queue<int> &result_queue) {
    CounterState state;
    while (!state.finish) {
        auto task = counter_queue.pop();
        task(state);
    }
    ostringstream ss;
    ss << "msg: " << state.msg << '\n';
    ss << "counter: " << state.counter << '\n';
    cout << ss.str();
    result_queue.push(state.counter);
}

void compute(mt_queue<function<void(CounterState &)>> &counter_queue, int beg, int end) {
    for (int i = beg; i < end; ++i) {
        counter_queue.push([i] (CounterState &state) {
            state.counter += i;
        });
    }
    counter_queue.push([] (CounterState &state) {
        state.msg += "OK";
    });
}

int main() {
    mt_queue<function<void(CounterState &)>> counter_queue;
    mt_queue<int> result_queue;

    vector<jthread> compute_pool;
    vector<jthread> counter_pool;
    for (int i = 0; i < 10000; i += 1000) {
        compute_pool.push_back(jthread(compute, std::ref(counter_queue), i, i + 1000));
    }
    counter_pool.push_back(jthread(counter_thread, std::ref(counter_queue), std::ref(result_queue)));
    counter_pool.push_back(jthread(counter_thread, std::ref(counter_queue), std::ref(result_queue)));
    counter_pool.push_back(jthread(counter_thread, std::ref(counter_queue), std::ref(result_queue)));
    for (auto &&t: compute_pool) {
        t.join();
    }
    counter_queue.push([] (CounterState &state) {
        state.finish = true;
    });
    counter_queue.push([] (CounterState &state) {
        state.finish = true;
    });
    counter_queue.push([] (CounterState &state) {
        state.finish = true;
    });
    for (auto &&t: counter_pool) {
        t.join();
    }
    int result = 0;
    for (int i = 0; i < 3; i++) {
        result += result_queue.pop();
    }
    cout << result << endl;
    return 0;
}
