#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

__global__ void naive_scan_kernel(int* d_out, const int* d_in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) return;

    int sum = 0;
    for (int i = 0; i < tid; ++i) {
        sum += d_in[i];
    }

    d_out[tid] = sum;
}

void naive_scan(const std::vector<int>& input, std::vector<int>& output) {
    int* d_in = nullptr;
    int* d_out = nullptr;
    int n = input.size();

    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_in, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    naive_scan_kernel<<<blocks, threads>>>(d_out, d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(output.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    std::vector<int> input = {3, 1, 7, 0, 4, 1, 6, 3};
    std::vector<int> output(input.size());

    naive_scan(input, output);

    std::cout << "Naive GPU Prefix Sum:\n";
    std::cout << "Input: ";
    for (int x : input) std::cout << x << " ";
    std::cout << "\nOutput: ";
    for (int x : output) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
