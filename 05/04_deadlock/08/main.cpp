#include <iostream>
#include <mutex>

std::recursive_mutex mtx1;

void other() {
    mtx1.lock();
    // do something
    mtx1.unlock();
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
