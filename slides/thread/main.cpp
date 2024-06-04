#include "mtqueue.hpp"
#include <iostream>
#include <thread>
using namespace std;

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
    jthread th1(t1);
    jthread th2(t2);
    this_thread::sleep_for(1.5s);
    th1.join();
    th2.join();
    return 0;
}
