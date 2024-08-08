#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <vector>

#define CHECK_CUDA(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s: %s\n", \
                __FILE__, __LINE__, \
                #expr, cudaGetErrorString(err)); \
        abort(); \
    } \
} while (0)

// 用 CUDA 实现分子动力学吧！

struct Atom {
    glm::vec3 r;
    glm::vec3 v;
    glm::vec3 F;
    float m;
};

std::vector<Atom> atoms;

int main() {
    float dt = 0.01f;
    for (auto &atom: atoms) {
        atom.v += atom.F / atom.m * dt / 2.0f;
    }
    for (auto &atom: atoms) {
        atom.r += atom.v * dt;
    }
    for (auto &atom: atoms) {
        atom.F = {};
    }
    for (auto &atom: atoms) {
        atom.v += atom.F / atom.m * dt / 2.0f;
    }
}
