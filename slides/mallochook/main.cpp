#include <thread>
#include <vector>
#include <chrono>

using namespace std::chrono_literals;

int main() {
    std::vector<int> a;
    for (int i = 0; i < 100; i++) {
        a.push_back(i);
    }
}
