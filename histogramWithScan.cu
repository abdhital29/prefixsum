#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

// Kernel to compute per-bin flags
__global__ void bin_flags(const int* input, int* bin_matrix, int n, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int val = input[tid];
    if (val < num_bins)
        bin_matrix[val * n + tid] = 1;  // Each bin has a row
}

// Kernel to reduce each bin's row using serial scan (for simplicity)
__global__ void reduce_bins(const int* bin_matrix, int* histogram, int n, int num_bins) {
    int bin = blockIdx.x;
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += bin_matrix[bin * n + i];
    }
    histogram[bin] = sum;
}

void histogram_scan(const std::vector<int>& input, std::vector<int>& histogram, int num_bins) {
    int n = input.size();

    int* d_input = nullptr;
    int* d_bin_matrix = nullptr;
    int* d_histogram = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_bin_matrix, n * num_bins * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_histogram, num_bins * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_bin_matrix, 0, n * num_bins * sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    bin_flags<<<blocks, threads>>>(d_input, d_bin_matrix, n, num_bins);
    reduce_bins<<<num_bins, 1>>>(d_bin_matrix, d_histogram, n, num_bins);

    histogram.resize(num_bins);
    CHECK_CUDA(cudaMemcpy(histogram.data(), d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_bin_matrix);
    cudaFree(d_histogram);
}
