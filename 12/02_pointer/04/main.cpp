#include <cstdio>
#include <cstdint>

int func(int& second) {
    second = 2;
    return 1;
}

int main() {
    int second;
    int first = func(second);
    printf("result: %d, %d\n", first, second);
    return 0;
}
