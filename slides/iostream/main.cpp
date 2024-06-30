#include <cstdio>
#include <thread>

using namespace std;

int main() {
    printf("请输入数字：");
    int x;
    scanf("%d", &x);
    int res = x * x;
    printf("%d的平方为：%d\n", x, res);
    return 0;
}
