#include <cstdio>
#include <cstdlib>
#include <vector>

struct C {
    C() {
        printf("分配内存!\n");
    }

    ~C() {
        printf("释放内存!\n");
    }
};

int main() {
    std::vector<C> p(1);

    if (rand() != 0) {
        printf("出了点小状况……\n");
        return 1;  // 自动释放 p
    }

    return 0;  // 自动释放 p
}
