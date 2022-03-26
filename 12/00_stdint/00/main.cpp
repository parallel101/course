#include <cstdio>
#include <cstdint>
#include <type_traits>

int main() {
    static_assert(std::is_same<decltype(0x7fffffff), int>::value, "小彭老师的断言");
    static_assert(std::is_same<decltype(0xffffffff), unsigned int>::value, "小彭老师的断言");
    static_assert(std::is_same<decltype(0x100000000), int64_t>::value, "小彭老师的断言");
    static_assert(std::is_same<decltype(0xffffffff'ffffffff), uint64_t>::value, "小彭老师的断言");
    return 0;
}
