#include <cstdio>

#ifdef _MSC_VER
__declspec(dllexport)
#endif
void say_hello() {
    printf("Hello, world!\n");
}
