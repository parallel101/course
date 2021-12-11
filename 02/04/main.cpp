#include <cstdlib>
#include <iostream>

struct Vector {
    size_t m_size;
    int *m_data;

    Vector(size_t n) {    // 构造函数 - 对象初始化时调用
        m_size = n;
        m_data = (int *)malloc(n * sizeof(int));
    }

    ~Vector() {           // 解构函数 - 对象销毁时调用
        free(m_data);
    }

    size_t size() {
        return m_size;
    }

    void resize(size_t size) {
        m_size = size;
        m_data = (int *)realloc(m_data, m_size);
    }

    int &operator[](size_t index) {  // 当 v[index] 时调用
        return m_data[index];
    }
};

int main() {
    Vector v(2);

    v[0] = 4;
    v[1] = 3;

    v.resize(4);

    v[2] = 2;
    v[3] = 1;

    int sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i];
    }

    std::cout << sum << std::endl;
}
