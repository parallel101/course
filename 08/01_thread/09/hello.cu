#include <cstdio>
#include <cuda_runtime.h>

__device__ void say_hello() {  // 定义
    printf("Hello, world!\n");
}
