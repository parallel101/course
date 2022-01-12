#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ticktock.h"

template <class T>
void quick_sort(T *data, size_t size) {
    if (size < 1)
        return;
    size_t mid = std::hash<size_t>{}(size);
    mid ^= std::hash<void *>{}(static_cast<void *>(data));
    mid %= size;
    std::swap(data[0], data[mid]);
    T pivot = data[0];
    size_t left = 0, right = size - 1;
    while (left < right) {
        while (left < right && !(data[right] < pivot))
            right--;
        if (left < right)
            data[left++] = data[right];
        while (left < right && data[left] < pivot)
            left++;
        if (left < right)
            data[right--] = data[left];
    }
    data[left] = pivot;
    quick_sort(data, left);
    quick_sort(data + left + 1, size - left - 1);
}

int main() {
    size_t n = 1<<24;
    std::vector<int> arr(n);
    std::generate(arr.begin(), arr.end(), std::rand);
    TICK(quick_sort);
    quick_sort(arr.data(), arr.size());
    TOCK(quick_sort);
    return 0;
}
