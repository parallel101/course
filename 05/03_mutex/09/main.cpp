#include <cstdio>
#include <mutex>

std::timed_mutex mtx1;

int main() {
    if (mtx1.try_lock_for(std::chrono::milliseconds(500)))
        printf("succeed\n");
    else
        printf("failed\n");

    if (mtx1.try_lock_for(std::chrono::milliseconds(500)))
        printf("succeed\n");
    else
        printf("failed\n");

    mtx1.unlock();
    return 0;
}
