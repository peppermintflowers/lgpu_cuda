#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    constexpr float ALPHA = 0.01;
    int thread_i = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread_i < N){
        float val = input[thread_i];
        if(val > 0){
            output[thread_i] = val;
        }
        else{
            output[thread_i] = ALPHA * val;
        }
    } 
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
