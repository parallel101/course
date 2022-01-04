#include <iostream>
#include <mutex>

std::mutex mtx1;

/// NOTE: please lock mtx1 before calling other()
void other() {
    // do something
}

void func() {
    mtx1.lock();
    other();
    mtx1.unlock();
}

int main() {
    func();
    return 0;
}
