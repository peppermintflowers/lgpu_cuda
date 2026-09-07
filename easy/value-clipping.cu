#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int thread_i = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread_i < N){
        float val = input[thread_i];
        if(val<=hi && val>=lo){
            output[thread_i] = val;
        }
        else if(val>hi){
            output[thread_i]=hi;
        }
        else{
            output[thread_i]=lo;
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
