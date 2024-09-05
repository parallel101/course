__global__ void kernel(int *sum, int *arr) {
    int i = threadIdx.x;
    atomicAdd(sum, arr[i]);
    __syncwarp();
}
