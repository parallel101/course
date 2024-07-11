#include <print>
#include <cstddef>

struct Vector {
private:
    int *m_data;
    size_t m_size;

public:
    Vector() : m_data(new int[4]), m_size(4) {}

    void setSize(size_t newSize) {
        m_size = newSize;
        delete[] m_data;
        m_data = new int[newSize];
    }

    int *data() const {
        return m_data;
    }

    size_t size() const {
        return m_size;
    }
};

int main() {
    Vector v;
    v.setSize(14);
    v.setSize(11);
    return 0;
}
