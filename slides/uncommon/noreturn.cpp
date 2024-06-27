#include <iostream>

int func() {
    std::cout << "进入 func() 内部\n";
    // 漏写 return 是未定义行为！不一定会报错
}

int main() {
    std::cout << "正在调用 func()\n";
    int res = func();
    std::cout << "func() 返回了" << res << '\n';
    return 0;
}

// GCC 用户可以通过 -Werror=return-type 选项将漏写 return 强制转化为报错
