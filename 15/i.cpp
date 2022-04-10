#include <cstdio>
#include <string>
#include <functional>
#include <map>

extern std::map<std::string, std::function<void()>> functab;

static void catFunc() {
    printf("Meow~\n");
}
static int defCat = (functab.emplace("cat", catFunc), 0);

static void dogFunc() {
    printf("Woah~\n");
}
static int defDog = (functab.emplace("dog", dogFunc), 0);

std::map<std::string, std::function<void()>> functab;

int main() {
    functab.at("cat")();  // equivalant to catFunc()
    functab.at("dog")();  // equivalant to dogFunc()
    return 0;
}
