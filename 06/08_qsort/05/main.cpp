#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tbb/parallel_sort.h>
#include "ticktock.h"

int main() {
    size_t n = 1<<24;
    std::vector<int> arr(n);
    std::generate(arr.begin(), arr.end(), std::rand);
    TICK(tbb_parallel_sort);
    tbb::parallel_sort(arr.begin(), arr.end(), std::less<int>{});
    TOCK(tbb_parallel_sort);
    return 0;
}
