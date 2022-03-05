#include <cstdio>

int main() {
#ifdef WITH_TBB
    printf("TBB enabled!\n");
#endif
    printf("Hello, world!\n");
}
