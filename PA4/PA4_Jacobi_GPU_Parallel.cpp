#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <omp.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

constexpr double MAX_RESIDUAL = 1.e-8;
constexpr int N = 1000;

using namespace std;

void jacobi_gpu_teams_simd(double* T, double* T_new, int max_iterations, int size) {
    int iteration = 0;
    double residual = 1.e6;

    while (residual > MAX_RESIDUAL && iteration < max_iterations) {
        residual = 0.0;

        #pragma omp target teams distribute parallel for simd collapse(2) map(to: T[0:size*size]) map(from: T_new[0:size*size])
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                T_new[i * size + j] = 0.25 * (
                    T[(i + 1) * size + j] + T[(i - 1) * size + j] +
                    T[i * size + (j + 1)] + T[i * size + (j - 1)]
                );
            }
        }

        #pragma omp target teams distribute parallel for simd reduction(max: residual) collapse(2) map(tofrom: T[0:size*size]) map(to: T_new[0:size*size])
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
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <max_iterations>" << endl;
        return 1;
    }

    int max_iterations = stoi(argv[1]);
    int size = N + 2;

    double* T = new double[size * size]();
    double* T_new = new double[size * size]();

    for (int i = 0; i < size; i++) {
        T[i * size + 0] = 1.0;
        T[i * size + (size - 1)] = 1.0;
    }

    auto start = chrono::high_resolution_clock::now();
    jacobi_gpu_teams_simd(T, T_new, max_iterations, size);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    cout << "Time taken for GPU Teams with SIMD: " << elapsed.count() << " s, N=" << N
         << ", Iterations=" << max_iterations << endl;

    delete[] T;
    delete[] T_new;

    return 0;
}
