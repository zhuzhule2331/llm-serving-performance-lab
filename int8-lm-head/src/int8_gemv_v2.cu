#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// INT8 Weight-Only GEMV V2
// 4 Warps / Block
// 1 Warp / Row
// Input:  FP32 [N]
// Weight: INT8 [M, N]
// Scale:  FP32 [M]
// Output: FP32 [M]

__global__ void int8_gemv_v2(
    const int8_t* __restrict__ Wq,
    const float* __restrict__ x,
    const float* __restrict__ scales,
    float* __restrict__ y,
    int M,
    int N
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int row = blockIdx.x * 4 + warp_id;

    if (row >= M) return;

    float sum = 0.0f;

    for (int j = lane; j < N; j += 32) {
        sum += static_cast<float>(
            Wq[static_cast<size_t>(row) * N + j]
        ) * x[j];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(
            0xffffffffu, sum, offset
        );
    }

    if (lane == 0) {
        y[row] = sum * scales[row];
    }
}
