#include <cuda_runtime.h>
#include <cstdint>

// INT8 Weight-Only GEMV
// 1 Block = 1 Warp = 32 Threads
// 1 Warp 负责计算矩阵的一行
// Wq: [M, N] INT8
// x:  [N] FP32
// scales: [M] FP32
// y: [M] FP32

__global__ void int8_gemv_v1(
    const int8_t* __restrict__ Wq,
    const float* __restrict__ x,
    const float* __restrict__ scales,
    float* __restrict__ y,
    int M,
    int N
) {
    // 1. 获取行号和线程编号
    int row = blockIdx.x;
    int lane = threadIdx.x;

    if (row >= M) return;

    // 2. 每个线程计算部分点积
    float sum = 0.0f;

    for (int j = lane; j < N; j += blockDim.x) {
        sum += static_cast<float>(Wq[(size_t)row * N + j])
             * x[j];
    }

    // 3. Warp Shuffle Reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(
            0xffffffffu,
            sum,
            offset
        );
    }

    // 4. Lane 0 写回结果
    if (lane == 0) {
        y[row] = sum * scales[row];
    }
}
