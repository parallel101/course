#include <iostream>
#include <vector>
#include <set>

int main() {
    std::vector<int> arr = {1, 4, 2, 8, 5, 7, 1, 4};
    std::set<int> visited;
    auto dfs = [&] (auto const &dfs, int index) -> void {
        if (visited.find(index) == visited.end()) {
            visited.insert(index);
            std::cout << index << std::endl;
            int next = arr[index];
            dfs(dfs, next);
        }
    };
    dfs(dfs, 0);
    return 0;
}
