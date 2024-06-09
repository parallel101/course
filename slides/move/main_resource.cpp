#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace std;

struct Resource {
private:
    struct Self {
        void *p;

        Self() {
            puts("分配资源");
            p = malloc(1);
        }

        Self(Self const &that) {
            puts("复制资源");
            p = malloc(1);
        }

        ~Self() {
            puts("释放资源");
            free(p);
        }
    };

    shared_ptr<Self> self;

    Resource(shared_ptr<Self> self_) {
        self = self_;
    }

public:
    Resource() {
        self = make_shared<Self>();
    }

    Resource(Resource const &) = default;

    Resource clone() const {
        return make_shared<Self>(*self);
    }

    void speak() const {
        printf("旺旺，我的资源句柄是 %p\n", self->p);
    }
};

void func(Resource x) {
    x.speak();
    auto y = x.clone();
    y.speak();
}

int main() {
    auto x = Resource();
    func(x);
}
