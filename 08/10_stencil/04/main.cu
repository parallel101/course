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

template <int iters, int blockSize>
__global__ void parallel_jacobi(float *out, float const *in, int nx, int ny) {
    int blockX = blockIdx.x;
    int blockY = blockIdx.x;
    int threadX = threadIdx.x;
    int threadY = threadIdx.x;
    int globalX = blockX * blockSize + threadX;
    int globalY = blockY * blockSize + threadY;

    auto at = [in, nx, ny] (int x, int y) -> float & {
        return in[std::min(std::max(x, 0), nx - 1) + nx * std::min(std::max(y, 0), ny - 1)];
    };

    __shared__ mem[2][blockSize + 2 * iters][blockSize + 2 * iters];
    mem[iters + threadY][iters + threadX] = at(globalX, globalY); 

    if (threadX < iters) {
        mem[0][iters + threadY][threadX] = at(blockX * blockSize - iters + threadX, blockY * blockSize + threadY);  // X-
        mem[0][threadX][iters + threadY] = at(blockX * blockSize + threadY, blockY * blockSize - iters + threadX);  // Y-
        mem[0][iters + threadY][iters + blockSize + threadX] = at(blockX * blockSize + iters + blockSize + threadX, blockY * blockSize + threadY);  // X+
        mem[0][iters + blockSize + threadX][iters + threadY] = at(blockX * blockSize + threadY, blockY * blockSize + iters + blockSize + threadX);  // Y+
    }

    mem[!phase][y][x] =
        ( mem[phase][iters + y + 1][iters + x]
        + mem[phase][iters + y - 1][iters + x]
        + mem[phase][iters + y][iters + x + 1]
        + mem[phase][iters + y][iters + x - 1]
        ) / 4;

    if (globalX >= nx || globalY >= ny)
        out[globalY * nx + globalX] = res;
}

int main() {
    std::vector<float, CudaAllocator<float>> in;
    std::vector<float, CudaAllocator<float>> out;

    auto [nx, ny, comp] = read_image(in, "original.jpg");
    out.resize(in.size());

    TICK(parallel_jacobi);

    constexpr int iters = 4;
    for (int step = 0; step < 256; step += iters) {
        parallel_jacobi<iters, 32><<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>
            (out.data(), in.data(), nx, ny);
        std::swap(out, in);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_jacobi);

    write_image(in, nx, ny, 1, "/tmp/out.png");
    system("display /tmp/out.png &");
    return 0;
}
