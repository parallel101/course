#include <cstdio>
#include <cstdint>
#include <type_traits>

int main() {
    static_assert(std::is_same<decltype((int)3 + (short)3), int>::value, "小彭老师的断言");
    return 0;
}
