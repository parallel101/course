#include <iostream>
#include <atomic>

int main() {
    boolalpha(std::cout);
    std::atomic<int> counter;

    counter.store(2);

    int old = 1;
    bool equal = counter.compare_exchange_strong(old, 3);
    std::cout << "equal=" << equal << std::endl;  // false
    std::cout << "old=" << old << std::endl;  // 2

    int now = counter.load();
    std::cout << "cnt=" << now << std::endl;  // 3

    old = 2;
    equal = counter.compare_exchange_strong(old, 3);
    std::cout << "equal=" << equal << std::endl;  // true
    std::cout << "old=" << old << std::endl;  // 1

    now = counter.load();
    std::cout << "cnt=" << now << std::endl;  // 3

    return 0;
}
