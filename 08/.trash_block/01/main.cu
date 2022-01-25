#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void reverse(int *in, int *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int val = in[n - 1 - i];
    out[i] = val;
}

int main() {
    int n = 1<<26;
    std::vector<int, CudaAllocator<int>> in(n);
    std::vector<int, CudaAllocator<int>> out(n);

    for (int i = 0; i < n; i++) {
        in[i] = i;
    }

    TICK(reverse);
    reverse<<<n / 128, 128>>>(in.data(), out.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(reverse);

    for (int i = 0; i < n; i++) {
        if (out[i] != in[n - 1 - i]) {
            printf("Wrong At %d\n", i);
            return -1;
        }
    }
    printf("All Correct!\n");

    return 0;
}
