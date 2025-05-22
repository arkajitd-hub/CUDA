#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
#define BLOCK_SIZE 1024

// ✅ GPU Kernel for Reduction
__global__ void reduction(double* input, double* output, int len) {
    __shared__ double partialSum[2 * BLOCK_SIZE];  
    unsigned int t = threadIdx.x;
    size_t start = 2 * (size_t)blockIdx.x * BLOCK_SIZE;  

    // ✅ Load elements into shared memory
    partialSum[t] = (start + t < len) ? input[start + t] : 0.0;
    partialSum[t + BLOCK_SIZE] = (start + BLOCK_SIZE + t < len) ? input[start + BLOCK_SIZE + t] : 0.0;
    __syncthreads();

    // ✅ Perform reduction
    for (unsigned int stride = BLOCK_SIZE; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            partialSum[t] += partialSum[t + stride];
        }
    }

    if (t == 0) output[blockIdx.x] = partialSum[0];
}

// ✅ CPU Sum Function for Verification
double getSumCPU(vector<double>& array) {
    double sum = 0.0;
    for (double num : array) sum += num;
    return sum;
}

// ✅ Main Function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <N>" << endl;
        return 1;
    }

    int N = stoi(argv[1]);
    vector<double> array(N);

    // ✅ Initialize array with values from 1 to N
    for (int i = 0; i < N; i++) {
        array[i] = i + 1;
    }

    auto start = chrono::high_resolution_clock::now();
    double result = getSumCPU(array);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "N: " << N << ", CPU Time: " << elapsed.count() << "s, Result = " << result << endl;

    // ✅ Allocate GPU Memory
    double *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(double));
    cudaMalloc((void**)&d_output, (N / (2 * BLOCK_SIZE)) * sizeof(double));  // Output buffer size

    cudaMemcpy(d_input, array.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // ✅ Iterative GPU Execution
    int numElements = N;
    while (numElements > 1) {
        int gridSize = (numElements + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
        dim3 dimGrid(gridSize, 1, 1);
        dim3 dimBlock(BLOCK_SIZE, 1, 1);

        reduction<<<dimGrid, dimBlock>>>(d_input, d_output, numElements);
        cudaDeviceSynchronize();

        // ✅ Copy partial sums back into d_input for next iteration
        cudaMemcpy(d_input, d_output, gridSize * sizeof(double), cudaMemcpyDeviceToDevice);
        numElements = gridSize;
    }

    // ✅ Copy Final Result
    double final_GPU_Sum;
    cudaMemcpy(&final_GPU_Sum, d_input, sizeof(double), cudaMemcpyDeviceToHost);

    auto gpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_elapsed = gpu_end - end;

    cout << "GPU Result = " << final_GPU_Sum << endl;
    cout << "GPU Time = " << gpu_elapsed.count() << "s" << endl;
    cout << "Speedup = " << elapsed.count() / gpu_elapsed.count() << endl;

    // ✅ Free Memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
