#include <iostream>
#include <atomic>

int main() {
    std::atomic<int> counter;

    counter.store(0);

    int old = counter.exchange(3);
    std::cout << "old=" << old << std::endl;  // 0

    int now = counter.load();
    std::cout << "cnt=" << now << std::endl;  // 3

    return 0;
}
