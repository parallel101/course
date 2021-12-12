#include <cstdio>
#include <memory>

struct C {
    C() {
        printf("分配内存!\n");
    }

    ~C() {
        printf("释放内存!\n");
    }
};

int main() {
    std::unique_ptr<C> p = std::make_unique<C>();

    printf("提前释放……\n");
    p = nullptr;
    printf("……释放成功\n");

    return 0;  // p 不会再释放一遍
}
