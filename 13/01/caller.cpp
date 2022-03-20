#include "ticktock.h"

void func1(float *dst, float *src);
void func2(float *dst, float *src);

int main() {
    auto *src = new float[4096 * 3]{};
    auto *dst = new float[4096]{};
    TICK(func1);
    func1(dst, src);
    TOCK(func1);
    TICK(func2);
    func2(dst, src);
    TOCK(func2);
    return 0;
}
