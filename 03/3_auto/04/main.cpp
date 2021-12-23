#include <cstdio>
#include <string>
#include <map>

auto &product_table() {
    static std::map<std::string, int> instance;
    return instance;
}

int main() {
    product_table().emplace("佩奇", 80);
    product_table().emplace("妈妈", 100);
}
