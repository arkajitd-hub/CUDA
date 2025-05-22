#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

constexpr double MAX_RESIDUAL = 1.e-15;
constexpr int N = 1000; // Grid size

using namespace std;

void jacobi(double* T, double* T_new, int max_iterations, int size) {
    int iteration = 0;
    double residual = 1.e6;

    while (residual > MAX_RESIDUAL && iteration < max_iterations) {
        residual = 0.0;

        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                T_new[i * size + j] = 0.25 * (
                    T[(i + 1) * size + j] + T[(i - 1) * size + j] +
                    T[i * size + (j + 1)] + T[i * size + (j - 1)]
                );
            }
        }

        // Compute Residual & Update T
        double max_residual = 0.0;
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                double diff = fabs(T_new[i * size + j] - T[i * size + j]);
                max_residual = MAX(diff, max_residual);
                T[i * size + j] = T_new[i * size + j];
            }
        }

        residual = max_residual;
        iteration++;
    }

    cout << "Final Residual: " << residual << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <max_iterations>" << endl;
        return 1;
    }

    int max_iterations = stoi(argv[1]);
    int size = N + 2; // Include boundary

    // Allocate memory for 1D arrays
    double* T = new double[size * size]();
    double* T_new = new double[size * size]();

    // Set boundary conditions
    for (int i = 0; i < size; i++) {
        T[i * size + 0] = 1.0;       // Left boundary
        T[i * size + (size - 1)] = 1.0;   // Right boundary
    }

    auto start = chrono::high_resolution_clock::now();
    jacobi(T, T_new, max_iterations, size);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    cout << "Time taken for Serial: " << elapsed.count() << " s, N=" << N
         << ", Iterations=" << max_iterations << endl;

    // Free allocated memory
    delete[] T;
    delete[] T_new;

    return 0;
}
