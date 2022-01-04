#include <cstdio>
#include <mutex>

std::mutex mtx1;

int main() {
    if (mtx1.try_lock())
        printf("succeed\n");
    else
        printf("failed\n");

    if (mtx1.try_lock())
        printf("succeed\n");
    else
        printf("failed\n");

    mtx1.unlock();
    return 0;
}
