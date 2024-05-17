#include <unordered_map>
#include <string>
#include <string_view>
#include <utility>

using namespace std;

void slow() {
    unordered_map<string, int, std::hash<string>, std::equal_to<>> table;
    pair<const string, int> entry = {"answer", 42};
    table.insert(std::as_const(entry));
    const string_view key = "hello";
    bool res = std::equal_to<>()(key, entry.first);
    table.find(key);
}
