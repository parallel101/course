#include <vector>
#include <numeric>

int func() {
    std::vector<int> arr;
    for (int i = 1; i <= 100; i++) {
        arr.push_back(i);
    }
    return std::reduce(arr.begin(), arr.end());
}
