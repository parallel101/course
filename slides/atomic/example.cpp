#include "mtpool.hpp"
using namespace std;

struct Test {
    int data = 0;

    void entry(MTIndex<0>) {  // 0 号线程
        data = 123;
    }

    void entry(MTIndex<1>) {  // 1 号线程
        data = 456;
    }

    void teardown() {  // 全部线程退出后时执行，用于收拾战场，汇报战果
        MTTest::result = data;
    }
};

int main() {
    MTTest::runTest<Test>();
    return 0;
}
