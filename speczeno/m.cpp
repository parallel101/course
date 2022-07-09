#include <cstdio>
#include <string>
#include <functional>
#include <map>

std::map<std::string, std::function<void()>> &getFunctab();

static void catFunc() {
    printf("Meow~\n");
}
static int defCat = (getFunctab().emplace("cat", catFunc), 0);

static void dogFunc() {
    printf("Woah~\n");
}
static int defDog = (getFunctab().emplace("dog", dogFunc), 0);

std::map<std::string, std::function<void()>> &getFunctab() {
    static std::map<std::string, std::function<void()>> inst;
    return inst;
}

int main() {
    getFunctab().at("cat")();  // equivalant to catFunc()
    getFunctab().at("dog")();  // equivalant to dogFunc()
    return 0;
}
