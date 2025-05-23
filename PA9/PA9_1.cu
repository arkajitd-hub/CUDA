//============================================================================
// Name        : FFTW_P09.cpp
// Description : Modified for benchmark timing only FFT and IFFT (excluding data prep/output)
//============================================================================

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <chrono>

#include "WavFile.h"

#define BUFF_SIZE   16384
#define MAX_FREQ    48

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void filterFrequencyDomain(fftw_complex* data, int N, double sampleRate, double freqToFilter, int) {
    int target_bin = static_cast<int>(freqToFilter * N / sampleRate);
    int mirror_bin = N - target_bin;

    if (target_bin >= 0 && target_bin + 1 < N && mirror_bin - 1 >= 0) {
        data[target_bin][0] = data[target_bin][1] = 0.0;
        data[target_bin + 1][0] = data[target_bin + 1][1] = 0.0;
        data[mirror_bin][0] = data[mirror_bin][1] = 0.0;
        data[mirror_bin - 1][0] = data[mirror_bin - 1][1] = 0.0;
    }
}

int main(int argc, char *argv[]) {
    const char *wavfile;
    if (argc != 2) {
        fprintf(stderr, "usage: %s <input.wav>\n", argv[0]);
        exit(1);
    } else {
        wavfile = argv[1];
    }

    fftw_complex *h_fft_in, *h_fft_out_cpu, *h_ifft_out_cpu, *h_fft_out_gpu_temp;
    h_fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    h_fft_out_cpu = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    h_ifft_out_cpu = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    h_fft_out_gpu_temp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);

    fftw_plan plan_forward_cpu = fftw_plan_dft_1d(BUFF_SIZE, h_fft_in, h_fft_out_cpu, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_backward_cpu = fftw_plan_dft_1d(BUFF_SIZE, h_fft_out_cpu, h_ifft_out_cpu, FFTW_BACKWARD, FFTW_ESTIMATE);

    cufftHandle plan_forward_gpu, plan_backward_gpu;
    cufftDoubleComplex *d_fft_data;
    cufftDoubleComplex *h_ifft_out_gpu = (cufftDoubleComplex *)calloc(BUFF_SIZE, sizeof(cufftDoubleComplex));

    CUDA_CHECK(cudaMalloc((void**)&d_fft_data, BUFF_SIZE * sizeof(cufftDoubleComplex)));
    cufftPlan1d(&plan_forward_gpu, BUFF_SIZE, CUFFT_Z2Z, 1);
    cufftPlan1d(&plan_backward_gpu, BUFF_SIZE, CUFFT_Z2Z, 1);

    short sampleBuffer[BUFF_SIZE];
    WavInFile inFile(wavfile);
    double sampleRate = inFile.getSampleRate();
    double freqToFilter = 10000.0;

    long long total_cpu_us = 0, total_gpu_us = 0;
    int chunk_count = 0;

    while (!inFile.eof()) {
        size_t samplesRead = inFile.read(sampleBuffer, BUFF_SIZE);
        if (samplesRead == 0) break;
        chunk_count++;

        for (size_t i = 0; i < BUFF_SIZE; ++i) {
            h_fft_in[i][0] = (i < samplesRead) ? (double)sampleBuffer[i] : 0.0;
            h_fft_in[i][1] = 0.0;
        }

        auto start_cpu = std::chrono::high_resolution_clock::now();
        fftw_execute(plan_forward_cpu);
        filterFrequencyDomain(h_fft_out_cpu, BUFF_SIZE, sampleRate, freqToFilter, 2);
        fftw_execute(plan_backward_cpu);
        auto stop_cpu = std::chrono::high_resolution_clock::now();
        total_cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count();

        auto start_gpu = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(d_fft_data, h_fft_in, BUFF_SIZE * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
        cufftExecZ2Z(plan_forward_gpu, d_fft_data, d_fft_data, CUFFT_FORWARD);
        CUDA_CHECK(cudaMemcpy(h_fft_out_gpu_temp, d_fft_data, BUFF_SIZE * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
        filterFrequencyDomain(h_fft_out_gpu_temp, BUFF_SIZE, sampleRate, freqToFilter, 2);
        CUDA_CHECK(cudaMemcpy(d_fft_data, h_fft_out_gpu_temp, BUFF_SIZE * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
        cufftExecZ2Z(plan_backward_gpu, d_fft_data, d_fft_data, CUFFT_INVERSE);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto stop_gpu = std::chrono::high_resolution_clock::now();
        total_gpu_us += std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu).count();
    }

    printf("\nProcessed %d chunks.\n", chunk_count);
    printf("CPU FFT+IFFT time avg per chunk: %.4f ms\n", (double)total_cpu_us / chunk_count / 1000.0);
    printf("GPU FFT+IFFT time avg per chunk: %.4f ms\n", (double)total_gpu_us / chunk_count / 1000.0);

    fftw_destroy_plan(plan_forward_cpu);
    fftw_destroy_plan(plan_backward_cpu);
    fftw_free(h_fft_in);
    fftw_free(h_fft_out_cpu);
    fftw_free(h_ifft_out_cpu);
    fftw_free(h_fft_out_gpu_temp);
    cufftDestroy(plan_forward_gpu);
    cufftDestroy(plan_backward_gpu);
    cudaFree(d_fft_data);
    free(h_ifft_out_gpu);

    return 0;
}
