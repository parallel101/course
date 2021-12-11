#include <fstream>
#include <stdexcept>

int main() {
    std::ofstream fout("a.txt");
    fout << "有一种病" << std::endl;
    throw std::runtime_error("中道崩殂");
    fout << "叫 JavaBean" << std::endl;
    return 0;
}
