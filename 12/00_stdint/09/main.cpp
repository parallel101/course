#include <cstdio>
#include <cstdint>
#include <type_traits>

int main() {
    static_assert(std::is_same<decltype((unsigned int)3 + (int)3), unsigned int>::value, "小彭老师的断言");
    return 0;
}
