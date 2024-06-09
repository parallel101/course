#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace std;

struct Resource {
    void *p;

    Resource() {
        puts("分配资源");
        p = malloc(1);
    }

    Resource(Resource &&that) : p(that.p) {
        that.p = nullptr;  // 一定要把对方置空！
    }

    Resource(Resource const &) = delete;

    ~Resource() {
        if (p) {
            puts("释放资源");
            free(p);
        }
    }
};

void func(Resource x) {
}

int main() {
    auto x = Resource();
    func(std::move(x));
}
