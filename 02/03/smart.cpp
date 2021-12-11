#include <fstream>
#include <vector>
#include <cstdio>

int main() {
    std::ifstream f1("1.txt");
    if (checkFileContent(f1)) {
        printf("bad file 1!\n");
        return 1;   // 自动释放 f1
    }

    std::ifstream f2("2.txt");
    if (checkFileContent(f2)) {
        printf("bad file 2!\n");
        return 1;   // 自动释放 f1, f2
    }

    vector<std::ifstream> files;
    files.push_back(std::ifstream("3.txt"));
    files.push_back(std::ifstream("4.txt"));
    files.push_back(std::ifstream("5.txt"));

    // files.clear();  // 提前释放 files（如果需要）

    return 0;   // 自动释放 f1, f2, files
}
