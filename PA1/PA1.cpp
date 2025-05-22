#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
#define ORDER 1000
#define AVAL 5.0
#define BVAL 7.0
int main(int argc, char* argv[]){
  int Pdim, Ndim, Mdim;
  int i,j,k;
  double tmpVal;
  Ndim = Mdim = Pdim = ORDER;
  vector<double> A(Ndim*Pdim, AVAL);
  vector<double> B(Pdim*Mdim, BVAL);
  vector<double> C(Mdim*Ndim, 0);
  int thread_nums = atoi(argv[1]);
  omp_set_num_threads(thread_nums);
  double start = omp_get_wtime();
  #pragma omp parallel for private(tmpVal, i, j, k)
  for(int i = 0; i<Ndim; i++){
    for(int j = 0; j<Mdim; j++){
      tmpVal = 0.0;
      for (k = 0; k < Pdim; k++) {
                tmpVal += A[i * Ndim + k] * B[k * Pdim + j];
            }
            C[i * Ndim + j] = tmpVal;
    }
  }
  double end = omp_get_wtime();
  double run_time = end - start;
  printf("%.6f,", run_time);
  return 0; 
}
