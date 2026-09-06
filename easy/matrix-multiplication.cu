#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
                                                int col = blockIdx.x * blockDim.x + threadIdx.x;
                                                int row = blockIdx.y * blockDim.y + threadIdx.y;

                                                if(row<M && col<K){
                                                    float sum=0.0f;
                                                    // C is MxK so striding by K for row major 
                                                    int cij = row*K + col;
                                                    // A is MxN so striding by N for row major
                                                    int aik = row*N;
                                                    for(int i=0; i<N; i++){
                                                        sum+= A[aik+i]*B[(K*i)+col];
                                                    }
                                                    C[cij] = sum;
                                                }
                                             }

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
