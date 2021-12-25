#include <iostream>
#include <vector>

int main() {
    std::vector<int> arr = {1, 4, 2, 8, 5, 7};
    int tofind = 5;
    int index = [&] {
        for (int i = 0; i < arr.size(); i++)
            if (arr[i] == tofind)
                return i;
        return -1;
    }();
    std::cout << index << std::endl;
    return 0;
}
