#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cassert>
#include <random>
#include <cmath>

using namespace std;

void convertToColumnMajor(const vector<vector<double>>& A, vector<double>& columnMajor) {
    int N = A.size();
    columnMajor.resize(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            columnMajor[j * N + i] = A[i][j];
}

vector<double> makeHilbertMatrix(int N) {
    vector<vector<double>> A(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i][j] = 1.0 / (i + j + 1);
    vector<double> columnMajor;
    convertToColumnMajor(A, columnMajor);
    return columnMajor;
}

vector<double> getResultVector(int N) {
    return vector<double>(N, 1.0);
}

vector<double> perturbVector(const vector<double>& vec) {
    vector<double> perturbed(vec);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.01, 1.0);
    for (double& v : perturbed) {
        v += dis(gen);
    }
    return perturbed;
}

double computeNorm(const vector<double>& vec) {
    double norm = 0.0;
    for (double val : vec) norm += val * val;
    return sqrt(norm);
}

int main(int argc, char* argv[]) {
    int N = stoi(argv[1]);
    const int lda = N;
    const int ldb = N;
    const bool pivot_on = false;

    vector<double> A = makeHilbertMatrix(N);
    vector<double> B1 = getResultVector(N);
    vector<double> B2 = perturbVector(B1);
    vector<double> X1(N), X2(N);

    cusolverDnHandle_t cusolverH;
    cudaStream_t stream;
    cudaEvent_t start, stop;
    float elapsed_ms;
    cout << "N - " << N << endl;
    double *d_A = nullptr, *d_B = nullptr, *d_work = nullptr;
    int *d_Ipiv = nullptr, *d_info = nullptr;
    int lwork = 0;

    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_A, sizeof(double) * N * N);
    cudaMalloc(&d_B, sizeof(double) * N);
    cudaMalloc(&d_info, sizeof(int));
    if (pivot_on) cudaMalloc(&d_Ipiv, sizeof(int) * N);

    cudaMemcpy(d_A, A.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B1.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    cusolverDnDgetrf_bufferSize(cusolverH, N, N, d_A, lda, &lwork);
    cudaMalloc(&d_work, sizeof(double) * lwork);

    
    cudaEventRecord(start);
    cusolverDnDgetrf(cusolverH, N, N, d_A, lda, d_work, d_Ipiv, d_info);
    cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_A, lda, d_Ipiv, d_B, ldb, d_info);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cout << "Time for LU + solve B1: " << elapsed_ms << " ms" << endl;

    cudaMemcpy(X1.data(), d_B, sizeof(double) * N, cudaMemcpyDeviceToHost);

    
    cudaMemcpy(d_B, B2.data(), sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_A, lda, d_Ipiv, d_B, ldb, d_info);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cout << "Time for solve B2 only: " << elapsed_ms << " ms" << endl;

    cudaMemcpy(X2.data(), d_B, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // check correctness results look skewed
    cout << "Norm of X1: " << computeNorm(X1) << endl;
    cout << "Norm of X2: " << computeNorm(X2) << endl;

    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_info);
    cudaFree(d_work);
    if (pivot_on) cudaFree(d_Ipiv);

    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();

    return 0;
}
