#include <cstdlib>
#include <iostream>
#include <cstring>

struct Vector {
    size_t m_size;
    int *m_data;

    Vector(size_t n) {
        m_size = n;
        m_data = (int *)malloc(n * sizeof(int));
    }

    ~Vector() {
        free(m_data);
    }

    Vector(Vector const &other) {
        m_size = other.m_size;
        m_data = (int *)malloc(m_size * sizeof(int));
        memcpy(m_data, other.m_data, m_size * sizeof(int));
    }

    Vector &operator=(Vector const &other) {
        this->~Vector();             // 先销毁现有的
        new (this) Vector(other);    // 再重新构造（placement new）
        return *this;                // 支持连等号：v1 = v2 = v3
    }

    size_t size() {
        return m_size;
    }

    void resize(size_t size) {
        m_size = size;
        m_data = (int *)realloc(m_data, m_size);
    }

    int &operator[](size_t index) {
        return m_data[index];
    }
};

int main() {
    Vector v1(32);
    Vector v2(64);

    v2 = v1;

    return 0;
}
