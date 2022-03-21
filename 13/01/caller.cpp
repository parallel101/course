#include "ticktock.h"

void func1(float *dst, float *src, int n);
void func2(float *dst, float *src, int n);
void func3(float *dst, float *src, int n);

int main() {
    int n = 1<<13;
    auto *src = new float[n * 3]{};
    auto *dst = new float[n]{};
    func1(dst, src, n);
    func2(dst, src, n);
    func3(dst, src, n);
    TICK(func1);
    func1(dst, src, n);
    TOCK(func1);
    TICK(func2);
    func2(dst, src, n);
    TOCK(func2);
    TICK(func3);
    func3(dst, src, n);
    TOCK(func3);
    return 0;
}
