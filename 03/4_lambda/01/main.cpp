#include <iostream>
#include <time.h>

void myfunc() {
    for (int i = 0; i < 1000000; i++) {
        printf("Test!\n");
    }
}

int main() {
    auto t0 = time(NULL);
    myfunc();
    auto t1 = time(NULL);
    auto dt = t1 - t0;
    std::cout << "Time elapsed: " << dt << "s" << std::endl;
    return 0;
}
