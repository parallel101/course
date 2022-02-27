#include <cstdio>
#ifdef WITH_TBB
#include <tbb/parallel_for.h>
#endif

int main() {
#ifdef WITH_TBB
    tbb::parallel_for(0, 4, [&] (int i) {
#else
    for (int i = 0; i < 4; i++) {
#endif
        printf("hello, %d!\n", i);
#ifdef WITH_TBB
    });
#else
    }
#endif
}
