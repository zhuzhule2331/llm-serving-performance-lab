#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "int8_gemv_v2.cu"

void launch_int8_gemv(
    at::Tensor x,
    at::Tensor Wq,
    at::Tensor scales,
    at::Tensor y
) {
    int M = static_cast<int>(Wq.size(0));
    int N = static_cast<int>(Wq.size(1));

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream().stream();

    int8_gemv_v2<<<(M + 3) / 4, 128, 0, stream>>>(
        Wq.data_ptr<int8_t>(),
        x.data_ptr<float>(),
        scales.data_ptr<float>(),
        y.data_ptr<float>(),
        M,
        N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
