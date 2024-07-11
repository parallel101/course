#include <print>
#include <vector>

int main() {
    std::vector<int> v;
    std::println("{} {}", (void *)v.data(), v.size()); // v.getData(), v.getSize()
    v.resize(14); // v.setSize(14)
    std::println("{} {}", (void *)v.data(), v.size()); // v.getData(), v.getSize()
    v.resize(16); // v.setSize(16)
    std::println("{} {}", (void *)v.data(), v.size()); // v.getData(), v.getSize()
    return 0;
}
