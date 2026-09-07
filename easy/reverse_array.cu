#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int thread_i = blockDim.x * blockIdx.x + threadIdx.x;
    int reversals = N/2;
    if(thread_i < reversals){
        float temp = input[N-1-thread_i];
        input[N-1-thread_i] = input[thread_i];
        input[thread_i] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
