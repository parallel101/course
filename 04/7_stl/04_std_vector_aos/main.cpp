#include <vector>

struct MyVec {
    float x;
    float y;
    float z;
};

std::vector<MyVec> a;

void func() {
    for (std::size_t i = 0; i < a.size(); i++) {
        a[i].x *= a[i].y;
    }
}
