#include <memory>

struct C {
    std::shared_ptr<C> m_child;
    C *m_parent;
};

int main() {
    auto parent = std::make_shared<C>();
    auto child = std::make_shared<C>();

    // 建立相互引用：
    parent->m_child = child;
    child->m_parent = parent.get();

    parent = nullptr;  // parent 会被释放。因为 child 指向他的是原始指针
    child = nullptr;   // child 会被释放。因为指向 child 的 parent 已经释放了

    return 0;
}
