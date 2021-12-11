#include <fstream>
#include <vector>
#include <cstdio>

int main() {
    std::ifstream f1("1.txt");
    if (checkFileContent(f1)) {
        printf("bad file 1!\n");
        f1.close();
        return 1;
    }

    std::ifstream f2("2.txt");
    if (checkFileContent(f2)) {
        printf("bad file 2!\n");
        f1.close();
        f2.close();
        return 1;
    }

    std::vector<std::ifstream> files;
    files.push_back(std::ifstream("3.txt"));
    files.push_back(std::ifstream("4.txt"));
    files.push_back(std::ifstream("5.txt"));

    for (auto &file: files)
        file.close();

    f1.close();
    f2.close();
    return 0;
}
