#include <cstdio>
#include <cstdint>

int main() {
    printf("uint8_t = %ld\n", sizeof(uint8_t));
    printf("uint16_t = %ld\n", sizeof(uint16_t));
    printf("uint32_t = %ld\n", sizeof(uint32_t));
    printf("uint64_t = %ld\n", sizeof(uint64_t));
    printf("uintptr_t = %ld\n", sizeof(uintptr_t));
    printf("int8_t = %ld\n", sizeof(int8_t));
    printf("int16_t = %ld\n", sizeof(int16_t));
    printf("int32_t = %ld\n", sizeof(int32_t));
    printf("int64_t = %ld\n", sizeof(int64_t));
    printf("intptr_t = %ld\n", sizeof(intptr_t));
    printf("pointer = %ld\n", sizeof(void *));
    return 0;
}
