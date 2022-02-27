#include <tbb/parallel_for.h>

int main() {
    tbb::parallel_for(0, 4, [&] (int i) {
        printf("hello, %d!\n", i);
    });
}
