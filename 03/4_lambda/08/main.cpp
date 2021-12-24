#include <vector>
#include <cstdio>

using namespace std;

using V = vector<int>;
void func(V &a, V &b, V &c) {
    c[0] = a[0];
    c[0] = b[0];
}

int main() {
    V a = {1}, b = {2};
    func(a, b, b);
    printf("%d\n", b[0]);
    return 0;
}
