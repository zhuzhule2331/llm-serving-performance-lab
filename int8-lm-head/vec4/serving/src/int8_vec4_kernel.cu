#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

__global__ void int8_gemv_v2_fallback(const int8_t* __restrict__ Wq,const float* __restrict__ x,const float* __restrict__ scales,float* __restrict__ y,int M,int N){
    const int warp_id=threadIdx.x/32, lane=threadIdx.x%32, row=blockIdx.x*4+warp_id;
    if(row>=M) return; float sum=0.0f;
    for(int j=lane;j<N;j+=32) sum += static_cast<float>(Wq[static_cast<size_t>(row)*N+j])*x[j];
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1) sum += __shfl_down_sync(0xffffffffu,sum,offset);
    if(lane==0) y[row]=sum*scales[row];
}

__global__ void int8_gemv_vec4(const int8_t* __restrict__ Wq,const float* __restrict__ x,const float* __restrict__ scales,float* __restrict__ y,int M,int N){
    const int warp_id=threadIdx.x/32, lane=threadIdx.x%32, row=blockIdx.x*4+warp_id;
    if(row>=M) return;
    const char4* __restrict__ w4=reinterpret_cast<const char4*>(Wq+static_cast<size_t>(row)*N);
    const float4* __restrict__ x4=reinterpret_cast<const float4*>(x);
    const int N4=N/4; float sum=0.0f;
    for(int j4=lane;j4<N4;j4+=32){
        const char4 q=w4[j4]; const float4 a=x4[j4];
        sum += static_cast<float>(q.x)*a.x;
        sum += static_cast<float>(q.y)*a.y;
        sum += static_cast<float>(q.z)*a.z;
        sum += static_cast<float>(q.w)*a.w;
    }
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1) sum += __shfl_down_sync(0xffffffffu,sum,offset);
    if(lane==0) y[row]=sum*scales[row];
}

void launch_int8_gemv_vec4(at::Tensor x,at::Tensor Wq,at::Tensor scales,at::Tensor y){
    const int M=static_cast<int>(Wq.size(0)), N=static_cast<int>(Wq.size(1));
    const auto* w=Wq.data_ptr<int8_t>(); const auto* xp=x.data_ptr<float>();
    cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    const bool ok=(N%128==0) && (reinterpret_cast<uintptr_t>(w)%alignof(char4)==0) && (reinterpret_cast<uintptr_t>(xp)%alignof(float4)==0);
    if(ok) int8_gemv_vec4<<<(M+3)/4,128,0,stream>>>(w,xp,scales.data_ptr<float>(),y.data_ptr<float>(),M,N);
    else int8_gemv_v2_fallback<<<(M+3)/4,128,0,stream>>>(w,xp,scales.data_ptr<float>(),y.data_ptr<float>(),M,N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
