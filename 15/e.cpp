#include <memory>
#include <cstdio>

using namespace std;

int main() {
    shared_ptr<int> p1 = make_shared<int>(42);
    shared_ptr<int> p2 = make_shared<int>(*p1);
    *p1 = 233;
    printf("%d\n", *p2);
    return 0;
}
