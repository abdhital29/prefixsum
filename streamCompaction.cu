#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

// Step 1: Mark non-zero flags
__global__ void mark_flags(const int* input, int* flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        flags[i] = input[i] != 0 ? 1 : 0;
}

// Step 2: Naive exclusive scan on flags
__global__ void scan_kernel(int* output, const int* input, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int sum = 0;
    for (int j = 0; j < i; ++j)
        sum += input[j];
    output[i] = sum;
}

// Step 3: Scatter elements
__global__ void scatter(const int* input, const int* flags, const int* scanned_flags, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i] == 1)
        output[scanned_flags[i]] = input[i];
}

// Host function
void stream_compaction(const std::vector<int>& input, std::vector<int>& compacted, int& valid_count) {
    int n = input.size();
    int* d_input = nullptr;
    int* d_flags = nullptr;
    int* d_scanned = nullptr;
    int* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_flags, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_scanned, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    mark_flags<<<blocks, threads>>>(d_input, d_flags, n);
    scan_kernel<<<blocks, threads>>>(d_scanned, d_flags, n);
    scatter<<<blocks, threads>>>(d_input, d_flags, d_scanned, d_output, n);

    // Get number of valid (non-zero) elements
    CHECK_CUDA(cudaMemcpy(&valid_count, &d_scanned[n - 1], sizeof(int), cudaMemcpyDeviceToHost));
    int last_flag;
    CHECK_CUDA(cudaMemcpy(&last_flag, &d_flags[n - 1], sizeof(int), cudaMemcpyDeviceToHost));
    valid_count += last_flag;

    compacted.resize(valid_count);
    CHECK_CUDA(cudaMemcpy(compacted.data(), d_output, valid_count * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_flags);
    cudaFree(d_scanned);
    cudaFree(d_output);
}
