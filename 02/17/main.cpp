#include <cstdio>
#include <memory>
#include <vector>

struct C {
    C() {
        printf("分配内存!\n");
    }

    ~C() {
        printf("释放内存!\n");
    }
};

std::vector<std::unique_ptr<C>> objlist;

void func(std::unique_ptr<C> p) {
    objlist.push_back(std::move(p));  // 进一步移动到 objlist
}

int main() {
    std::unique_ptr<C> p = std::make_unique<C>();
    printf("移交前：%p\n", p.get());  // 不为 null
    func(std::move(p));    // 通过移动构造函数，转移指针控制权
    printf("移交后：%p\n", p.get());  // null，因为移动会清除原对象
    return 0;
}
