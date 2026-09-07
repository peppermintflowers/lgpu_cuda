#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    //Here one thread points to a rgba set i.e. one pixel.
    //Using size_t to be able to handle large input sizes.
    size_t thread_i = (size_t) blockDim.x * blockIdx.x + threadIdx.x ;
    unsigned char inverter = 255;
    size_t total_rgbas = (size_t)width*height;
    if(thread_i < total_rgbas){
            image[4*thread_i] = inverter - image[4*thread_i];
            image[4*thread_i + 1] = inverter - image[4*thread_i + 1];
            image[4*thread_i + 2] = inverter - image[4*thread_i + 2];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
