#include <cstdio>
#include <cstdint>

int main() {
    float x = 1.0f;
    float* p = &x;
    printf("x = %f\n", x);
    *p = 3.14f;
    printf("x = %f\n", x);
    return 0;
}
