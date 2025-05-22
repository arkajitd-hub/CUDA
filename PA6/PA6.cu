#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define TILE_WIDTH 32 

using namespace std;


__global__ void matrixBasicCUDA(int *A, int *B, int *C, int m, int p, int n) {
    //get row, col index to operate on C matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int sum = 0.0;
        for (int k = 0; k < p; k++) {
            sum += A[row * p + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
__global__ void matrixTiledMultiply(int *A, int *B, int *C, int m, int p, int n) {
    __shared__ int aTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ int bTile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int sum = 0.0;

    for (int i = 0; i < (p - 1) / TILE_WIDTH + 1; i++) {
        if (row < m && i * TILE_WIDTH + tx < p) {
            aTile[ty][tx] = A[row * p + i * TILE_WIDTH + tx];
        } else {
            aTile[ty][tx] = 0.0;
        }

        if (col < n && i * TILE_WIDTH + ty < p) {
            bTile[ty][tx] = B[(i * TILE_WIDTH + ty) * n + col];
        } else {
            bTile[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += aTile[ty][k] * bTile[k][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}
void initializeMatrix(int *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<int>(rand()) / 20;
    }
}
int main(int argc, char **argv) {
    int m = stoi(argv[1]);
    int p = stoi(argv[2]);
    int n = m; 

    int *hA = new int[m * p];
    int *hB = new int[p * n];
    int *hC_basic = new int[m * n];
    int *hC_tiled = new int[m * n];
    srand(time(0));
    initializeMatrix(hA, m * p);
    initializeMatrix(hB, p * n);
    int *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(int) * m * p);
    cudaMalloc(&dB, sizeof(int) * p * n);
    cudaMalloc(&dC, sizeof(int) * m * n);

    cudaMemcpy(dA, hA, sizeof(int) * m * p, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(int) * p * n, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
    cudaEvent_t start, stop;
    float time_basic, time_tiled;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixBasicCUDA<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_basic, start, stop);
    cudaMemcpy(hC_basic, dC, sizeof(int) * m * n, cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    matrixTiledMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tiled, start, stop);
    cudaMemcpy(hC_tiled, dC, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    cout << "Matrix Size: " <<" m: " << m  << " p: " << p << " n: " << n  << endl;
    cout << "Basic Kernel Time: " << time_basic << " ms" << endl;
    cout << "Tiled Kernel Time: " << time_tiled << " ms" << endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC_basic;
    delete[] hC_tiled;

    return 0;
}
