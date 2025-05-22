#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <omp.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

constexpr double MAX_RESIDUAL = 1.e-8;
constexpr int N = 1000;

using namespace std;

void jacobi(double* T, double* T_new, int max_iterations, int size, int num_threads) {
    int iteration = 0;
    double residual = 1.e6;

    omp_set_num_threads(num_threads);

    while (residual > MAX_RESIDUAL && iteration < max_iterations) {
        residual = 0.0;

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                T_new[i * size + j] = 0.25 * (
                    T[(i + 1) * size + j] + T[(i - 1) * size + j] +
                    T[i * size + (j + 1)] + T[i * size + (j - 1)]
                );
            }
        }

        #pragma omp parallel for reduction(max : residual) collapse(2)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                double diff = fabs(T_new[i * size + j] - T[i * size + j]);
                residual = MAX(diff, residual);
                T[i * size + j] = T_new[i * size + j];
            }
        }

        iteration++;
    }

    cout << "Final Residual: " << residual << endl;
}

int main(int argc, char* argv[]) {
    
    int max_iterations = stoi(argv[1]);
    int size = N + 2;
    int num_threads = stoi(argv[2]);

    double* T = new double[size * size]();
    double* T_new = new double[size * size]();

    for (int i = 0; i < size; i++) {
        T[i * size + 0] = 1.0;
        T[i * size + (size - 1)] = 1.0;
    }

    auto start = chrono::high_resolution_clock::now();
    jacobi(T, T_new, max_iterations, size, num_threads);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    cout << "Time taken for CPU Parallel: " << elapsed.count() << " s, N=" << N
         << ", Iterations=" << max_iterations << ", Thread Count=" << num_threads << endl;

    delete[] T;
    delete[] T_new;

    return 0;
}
