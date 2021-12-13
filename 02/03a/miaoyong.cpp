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
