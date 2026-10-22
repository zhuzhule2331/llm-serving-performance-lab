import os
from pathlib import Path
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'build'; BUILD.mkdir(exist_ok=True)
print('[BUILD] TORCH_CUDA_ARCH_LIST =', os.environ['TORCH_CUDA_ARCH_LIST'])
print('[BUILD] output =', BUILD)
mod=load(
    name='int8_gemv_vec4_serving_ext',
    sources=[str(ROOT/'src/int8_vec4_binding.cpp'),str(ROOT/'src/int8_vec4_kernel.cu')],
    extra_cflags=['-O3'], extra_cuda_cflags=['-O3','-lineinfo'],
    build_directory=str(BUILD), verbose=True, with_cuda=True,
)
print('[BUILD] PASS:', mod.__file__)
