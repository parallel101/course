#include <print>

struct Point {
    double x;
    double y;

    Point operator+(Point const &other) const {
        return Point(x + other.x, y + other.y);
    }
};

int main() {
    Point a = Point{ .x = 1, .y = 2 }; // 等价于 Point{1, 2}
    Point b = Point{ .x = 2, .y = 3 }; // 等价于 Point{2, 3}
    Point c = a + b;
    std::println("{} {}", c.x, c.y);
    c.x = 1;
    return 0;
}
