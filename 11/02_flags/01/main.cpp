#include <cstdio>

int main() {
#ifdef MY_MACRO
    printf("MY_MACRO defined! value: %d\n", MY_MACRO);
#else
    printf("MY_MACRO not defined!\n");
#endif
}
