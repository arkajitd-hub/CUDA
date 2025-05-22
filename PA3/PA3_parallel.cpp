#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cmath>
using namespace std;

double function(double x){
	return acos(cos(x)/(1+2*cos(x)));
}

double simpsons_method_omp(double a, double b, int N,int numThreads)
{
    double h = (b - a) / N;
    double sum = function(a) + function(b);

    double oddSum = 0.0, evenSum = 0.0;

    #pragma omp parallel for num_threads(numThreads) reduction(+:oddSum)
    for (int i = 1; i < N; i += 2) { // Odd indices
        double x = a + i * h;
        oddSum += function(x);
    }

    #pragma omp parallel for num_threads(numThreads) reduction(+:evenSum)
    for (int i = 2; i < N; i += 2) { // Even indices
        double x = a + i * h;
        evenSum += function(x);
    }

    return (h / 3) * (sum + 4 * oddSum + 2 * evenSum);
}


int main(int argc, char* argv[]){
  int N = stoi(argv[1]);
  int num_threads = stoi(argv[2]);
  double b = M_PI/2;
  double a = 0;
  auto start = chrono::high_resolution_clock::now();
  double approx = simpsons_method_omp(a,b,N,num_threads);
  cout<< "Approximate " << approx << endl;
  double result = (5*(M_PI)*(M_PI)) / 24;
  double diff = abs(result - approx);
  cout << " Diff " << diff << endl;
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed = end - start;

    cout << "Optimized Parallel Execution Time for N=" << N << " with " << num_threads 
         << " threads: " << elapsed.count() << " sec\n";

    return 0;
}
