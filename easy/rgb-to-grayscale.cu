#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int thread_i = blockDim.x * blockIdx.x + threadIdx.x;
    int pixels = height * width;
    if(thread_i < pixels){
       output[thread_i] = 0.299 * input[3 * thread_i] + 0.587 * input[3 * thread_i + 1] + 0.114 * input[3 * thread_i + 2];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
