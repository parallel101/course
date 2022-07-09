#include <cstdio>
#include <type_traits>


int main() {
    if (std::is_signed<char>::value) {
        printf("你的 char 是有符号的，我猜你是 x86 架构\n");
    } else {
        printf("你的 char 是无符号的，我猜你是 arm 架构\n");
    }
    return 0;
}
