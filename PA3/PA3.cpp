#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
double function(double x){
  return acos(cos(x)/(1+2*cos(x)));
}
double findIntegral(int N, double b, double a){
    double h = (b - a) / N;
    double sum = function(a) + function(b);
    double oddSum = 0.0, evenSum = 0.0;

    for (int i = 1; i < N; i += 2) { // Odd indices
        double x = a + i * h;
        oddSum += function(x);
    }

    for (int i = 2; i < N; i += 2) { // Even indices
        double x = a + i * h;
        evenSum += function(x);
    }

    return (h / 3) * (sum + 4 * oddSum + 2 * evenSum);
}

int main(int argc, char* argv[]){
  int N = stoi(argv[1]);
  double b = M_PI/2;
  double a = 0;
  auto start = chrono::high_resolution_clock::now();
  double approx = findIntegral(N,b,a);
  cout<< "Approc " << approx << endl;
  double result = (5*(M_PI)*(M_PI)) / 24;
  double diff = abs(result - approx);
  cout << " Diff " << diff << endl;
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed = end - start;
  cout << "Serial Execution Time for N=" << N << " " << elapsed.count() << " sec\n";
  return 0;
}
