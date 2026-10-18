#include <torch/extension.h>
#include <climits>

void launch_int8_gemv(
    torch::Tensor x,
    torch::Tensor Wq,
    torch::Tensor scales,
    torch::Tensor y
);

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor Wq,
    torch::Tensor scales
) {
    TORCH_CHECK(
        x.is_cuda() && Wq.is_cuda() && scales.is_cuda(),
        "All tensors must be CUDA tensors"
    );

    TORCH_CHECK(
        x.device() == Wq.device() &&
        x.device() == scales.device(),
        "Tensors must be on the same GPU"
    );

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be FP32");
    TORCH_CHECK(Wq.dtype() == torch::kInt8, "Wq must be INT8");
    TORCH_CHECK(scales.dtype() == torch::kFloat32,
                "scales must be FP32");

    TORCH_CHECK(x.dim() == 1, "x must be [N]");
    TORCH_CHECK(Wq.dim() == 2, "Wq must be [M,N]");
    TORCH_CHECK(scales.dim() == 1, "scales must be [M]");

    const auto M = Wq.size(0);
    const auto N = Wq.size(1);

    TORCH_CHECK(M > 0 && N > 0, "M and N must be positive");
    TORCH_CHECK(M <= INT_MAX && N <= INT_MAX,
                "Dimensions exceed CUDA kernel limits");
    TORCH_CHECK(x.size(0) == N, "Input dimension mismatch");
    TORCH_CHECK(scales.size(0) == M, "Scale dimension mismatch");

    TORCH_CHECK(
        x.is_contiguous() &&
        Wq.is_contiguous() &&
        scales.is_contiguous(),
        "All tensors must be contiguous"
    );

    auto y = torch::empty({M}, x.options());

    launch_int8_gemv(x, Wq, scales, y);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "INT8 GEMV V2 forward");
}
