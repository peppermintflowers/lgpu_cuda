#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col_i = blockDim.x * blockIdx.x + threadIdx.x ;
    int row_i = blockDim.y * blockIdx.y + threadIdx.y;
    // row major format
    //access contiguous elements in row using input[cols * row_i + col_i]
    //access elements in col using input[rows * col_i + row_i]
    if(row_i < rows && col_i < cols){
        output[rows * col_i + row_i] = input[cols * row_i + col_i];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
