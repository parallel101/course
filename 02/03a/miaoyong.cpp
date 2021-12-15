struct {
    bool hit;
    Vec3 pos;
    Vec3 normal;
    float depth;
} intersect(Ray r) {
    ...
    return {true, r.origin, r.direction, 233.0f};
}

int main() {
    Ray r;
    auto hit = intersect(r);
    if (hit.hit) {
        r.origin = hit.pos;
        r.direction = hit.normal;
        ...
    }
}

void func(std::tuple<int, float, std::string> arg, std::vector<int> arr) {
    ...
}

int main() {
    func({1, 3.14f, "佩奇"}, {1, 4, 2, 8, 5, 7});
    // 等价于：
    func(std::tuple<int, float, std::string>(1, 3.14f, "佩奇"),
         std::vector<int>({1, 4, 2, 8, 5, 7}));
    // （C++17起）等价于：
    func(std::tuple(1, 3.14f, "佩奇"), std::vector({1, 4, 2, 8, 5, 7}));
}
