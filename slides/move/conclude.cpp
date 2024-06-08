#include <memory>
#include <vector>

// 掌管资源的类？四大函数全部删光！然后安心定义析构函数
struct Resource {
    Resource() {
        // 分配资源
    }

    Resource(Resource &&) = delete;
    Resource(Resource const &) = delete;            // 可省略不写
    Resource &operator=(Resource &&) = delete;      // 可省略不写
    Resource &operator=(Resource const &) = delete; // 可省略不写

    ~Resource() {
        // 释放资源
    }
};

// 如果这个类需要作为参数传递，需要移动怎么办？
std::unique_ptr<Resource> p = std::make_unique<Resource>();

// 每次都要 std::move 不方便，我想要随心所欲的浅拷贝，就像 Java 对象一样，怎么办？
std::shared_ptr<Resource> q = std::make_shared<Resource>();

// 不管理资源的类？那就都不用定义了！编译器会自动生成
struct Student {
    std::string name;
    int age;
    std::vector<int> scores;
    std::shared_ptr<Resource> bigHouse;

    // 编译器自动生成 Student 的拷贝构造函数为成员全部依次拷贝，不用你自己定义了
};
