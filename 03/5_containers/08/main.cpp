#include <iostream>
#include <optional>
#include <cmath>

std::optional<float> mysqrt(float x) {
    if (x >= 0.f) {
        return std::sqrt(x);
    } else {
        return std::nullopt;
    }
}

int main() {
    auto ret = mysqrt(-3.14f);
    if (ret.has_value()) {
        printf("成功！结果为：%f\n", *ret);
    } else {
        printf("失败！找不到平方根！\n");
    }
    return 0;
}
