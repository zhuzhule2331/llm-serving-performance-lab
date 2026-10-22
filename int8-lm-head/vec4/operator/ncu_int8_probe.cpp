#include <torch/extension.h>
#include <cstdint>

void launch_v2_probe(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y);
void launch_char4_probe(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y);
void launch_vec4_probe(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y);

static void validate(const at::Tensor& x, const at::Tensor& w,
                     const at::Tensor& s, const at::Tensor& y) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && s.is_cuda() && y.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.device() == w.device() && x.device() == s.device() && x.device() == y.device(), "device mismatch");
    TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat, "x/y must be FP32");
    TORCH_CHECK(w.scalar_type() == at::kChar, "Wq must be INT8");
    TORCH_CHECK(s.scalar_type() == at::kFloat, "scales must be FP32");
    TORCH_CHECK(x.dim() == 1 && y.dim() == 1 && w.dim() == 2 && s.dim() == 1, "invalid dimensions");
    TORCH_CHECK(w.size(0) == 151936 && w.size(1) == 4096, "only target shape [151936,4096] supported");
    TORCH_CHECK(x.size(0) == w.size(1) && y.size(0) == w.size(0) && s.size(0) == w.size(0), "shape mismatch");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && s.is_contiguous() && y.is_contiguous(), "must be contiguous");
}

void v2_into(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y) {
    validate(x,w,s,y);
    launch_v2_probe(x,w,s,y);
}
void char4_into(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y) {
    validate(x,w,s,y);
    TORCH_CHECK(w.size(1) % 128 == 0, "char4 kernel requires N divisible by 128");
    launch_char4_probe(x,w,s,y);
}
void vec4_into(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor y) {
    validate(x,w,s,y);
    TORCH_CHECK(w.size(1) % 128 == 0, "vec4 kernel requires N divisible by 128");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr<float>()) & 0xF) == 0,
                "x must be 16-byte aligned for float4 loads");
    launch_vec4_probe(x,w,s,y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("v2_into", &v2_into, "Original V2 into preallocated output");
    m.def("char4_into", &char4_into, "Weight-only char4 probe");
    m.def("vec4_into", &vec4_into, "char4 weight + float4 activation probe");
}
