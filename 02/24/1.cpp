#include <memory>

struct C {
    std::shared_ptr<C> m_child;
    std::weak_ptr<C> m_parent;
};

int main() {
    auto parent = std::make_shared<C>();
    auto child = std::make_shared<C>();

    // 建立相互引用：
    parent->m_child = child;
    child->m_parent = parent;

    parent = nullptr;  // parent 会被释放。因为 child 指向他的是 **弱引用**
    child = nullptr;   // child 会被释放。因为指向 child 的 parent 已经释放了

    return 0;
}
