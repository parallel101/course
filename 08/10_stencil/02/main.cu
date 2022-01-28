#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"
#include "stb_image.h"
#include "stb_image_write.h"

template <class A>
std::tuple<int, int, int> read_image(A &a, const char *path) {
    int nx = 0, ny = 0, comp = 0;
    unsigned char *p = stbi_load(path, &nx, &ny, &comp, 0);
    if (!p) {
        perror(path);
        exit(-1);
    }
    a.resize(nx * ny * comp);
    for (int c = 0; c < comp; c++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a[c * nx * ny + y * nx + x] = (1.f / 255.f) * p[(y * nx + x) * comp + c];
            }
        }
    }
    stbi_image_free(p);
    return {nx, ny, comp};
}

template <class A>
void write_image(A const &a, int nx, int ny, int comp, const char *path) {
    auto p = (unsigned char *)malloc(nx * ny * comp);
    for (int c = 0; c < comp; c++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                p[(y * nx + x) * comp + c] = std::max(0.f, std::min(255.f, a[c * nx * ny + y * nx + x] * 255.f));
            }
        }
    }
    int ret = 0;
    auto pt = strrchr(path, '.');
    if (pt && !strcmp(pt, ".png")) {
        ret = stbi_write_png(path, nx, ny, comp, p, 0);
    } else if (pt && !strcmp(pt, ".jpg")) {
        ret = stbi_write_jpg(path, nx, ny, comp, p, 0);
    } else {
        ret = stbi_write_bmp(path, nx, ny, comp, p);
    }
    free(p);
    if (!ret) {
        perror(path);
        exit(-1);
    }
}

template <int nblur, int blockSize>
__global__ void parallel_yblur(float *out, float const *in, int nx, int ny) {
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;
    if (x >= nx || y >= ny) return;
    float sum = 0;
    for (int i = 0; i < nblur; i++) {
        sum += in[std::min(y + i, ny - 1) * nx + x];
    }
    out[y * nx + x] = sum / nblur;
}

int main() {
    std::vector<float, CudaAllocator<float>> in;
    std::vector<float, CudaAllocator<float>> out;

    auto [nx, ny, comp] = read_image(in, "original.jpg");
    out.resize(in.size());

    TICK(parallel_yblur);
    parallel_yblur<32, 32><<<dim3((nx + 31) / 32, (ny + 31) / 32, 1), dim3(32, 32, 1)>>>
        (out.data(), in.data(), nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_yblur);

    write_image(out, nx, ny, 1, "/tmp/out.png");
    system("display /tmp/out.png &");
    return 0;
}
