#include <iostream>
#include <tuple>
#include <cmath>

std::tuple<bool, float> mysqrt(float x) {
    if (x >= 0.f) {
        return {true, std::sqrt(x)};
    } else {
        return {false, 0.0f};
    }
}

int main() {
    auto [success, value] = mysqrt(3.f);
    if (success) {
        printf("成功！结果为：%f\n", value);
    } else {
        printf("失败！找不到平方根！\n");
    }
    return 0;
}
