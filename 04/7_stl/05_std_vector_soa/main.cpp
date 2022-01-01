#include <vector>

struct MyVec {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

MyVec a;

void func() {
    for (std::size_t i = 0; i < a.x.size(); i++) {
        a.x[i] *= a.y[i];
    }
}
