#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// Exact baseline snapshot, kept unmodified.
#include "baseline_int8_gemv_v2.cu"

// Probe 1: vectorize only the INT8 weight load.
// This is intentionally kept because the previous NCU run showed that it
// reduces global-load instructions but makes x accesses stride by 16 B across
// lanes for each scalar x component, inflating L1/L2 sector traffic.
__global__ void int8_gemv_char4_probe(const int8_t* __restrict__ Wq,
                                      const float* __restrict__ x,
                                      const float* __restrict__ scales,
                                      float* __restrict__ y,
                                      int M, int N) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + warp;
    if (row >= M) return;

    float sum = 0.0f;
    const size_t row_start = static_cast<size_t>(row) * N;

    for (int j = lane * 4; j < N; j += 128) {
        const char4 q = *reinterpret_cast<const char4*>(Wq + row_start + j);
        sum += static_cast<float>(static_cast<int8_t>(q.x)) * x[j + 0];
        sum += static_cast<float>(static_cast<int8_t>(q.y)) * x[j + 1];
        sum += static_cast<float>(static_cast<int8_t>(q.z)) * x[j + 2];
        sum += static_cast<float>(static_cast<int8_t>(q.w)) * x[j + 3];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) y[row] = sum * scales[row];
}

// Probe 2 (data-driven follow-up): vectorize BOTH weight and activation loads.
// Mapping remains 1 warp / row and 4 warps / block.
// Per iteration, each lane consumes four consecutive Wq bytes and four
// consecutive x floats. Across the warp, both accesses are contiguous:
//   Wq:    32 lanes * 4 B  = 128 B
//   x:     32 lanes * 16 B = 512 B
// This preserves the original kernel's ideal coalescing instead of the
// scalar-x striding introduced by Probe 1.
__global__ void int8_gemv_char4_float4_probe(const int8_t* __restrict__ Wq,
                                             const float* __restrict__ x,
                                             const float* __restrict__ scales,
                                             float* __restrict__ y,
                                             int M, int N) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + warp;
    if (row >= M) return;

    float sum = 0.0f;
    const size_t row_start = static_cast<size_t>(row) * N;

    for (int j = lane * 4; j < N; j += 128) {
        const char4 q = *reinterpret_cast<const char4*>(Wq + row_start + j);
        const float4 xv = *reinterpret_cast<const float4*>(x + j);
        sum += static_cast<float>(static_cast<int8_t>(q.x)) * xv.x;
        sum += static_cast<float>(static_cast<int8_t>(q.y)) * xv.y;
        sum += static_cast<float>(static_cast<int8_t>(q.z)) * xv.z;
        sum += static_cast<float>(static_cast<int8_t>(q.w)) * xv.w;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) y[row] = sum * scales[row];
}

void launch_v2_probe(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int M = static_cast<int>(w.size(0));
    const int N = static_cast<int>(w.size(1));
    int8_gemv_v2<<<(M + 3) / 4, 128, 0, stream>>>(
        w.data_ptr<int8_t>(), x.data_ptr<float>(), s.data_ptr<float>(),
        y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_char4_probe(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int M = static_cast<int>(w.size(0));
    const int N = static_cast<int>(w.size(1));
    int8_gemv_char4_probe<<<(M + 3) / 4, 128, 0, stream>>>(
        w.data_ptr<int8_t>(), x.data_ptr<float>(), s.data_ptr<float>(),
        y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_vec4_probe(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int M = static_cast<int>(w.size(0));
    const int N = static_cast<int>(w.size(1));
    int8_gemv_char4_float4_probe<<<(M + 3) / 4, 128, 0, stream>>>(
        w.data_ptr<int8_t>(), x.data_ptr<float>(), s.data_ptr<float>(),
        y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
