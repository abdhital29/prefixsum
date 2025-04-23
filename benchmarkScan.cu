#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <numeric>
#include <cstdlib>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

extern void blelloch_scan(const std::vector<int>& input, std::vector<int>& output);

// CPU Scan (Exclusive)
void cpu_scan(const std::vector<int>& input, std::vector<int>& output) {
    output[0] = 0;
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// Naive GPU Kernel
__global__ void naive_scan_kernel(int* d_out, const int* d_in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int sum = 0;
    for (int i = 0; i < tid; ++i) {
        sum += d_in[i];
    }
    d_out[tid] = sum;
}

void naive_gpu_scan(const std::vector<int>& input, std::vector<int>& output) {
    int* d_in = nullptr;
    int* d_out = nullptr;
    int n = input.size();

    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_in, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naive_scan_kernel<<<blocks, threads>>>(d_out, d_in, n);
    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CHECK_CUDA(cudaMemcpy(output.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Naive GPU Time: " << milliseconds << " ms\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark(int n) {
    std::cout << "\n--- Benchmarking scan for n = " << n << " ---\n";
    std::vector<int> input(n), output_cpu(n), output_gpu(n);

    // Fill input with random data
    for (int& x : input) x = rand() % 10;

    // CPU timing
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_scan(input, output_cpu);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "CPU Time: " << cpu_ms << " ms\n";

    // Naive GPU
    naive_gpu_scan(input, output_gpu);

    // Verify correctness
    bool correct = (output_cpu == output_gpu);
    std::cout << "Results match? " << (correct ? "✅" : "❌") << "\n";

    // blelloch scan
    std::vector<int> output_blelloch(n);
    blelloch_scan(input, output_blelloch);
    std::cout << "Blelloch correct? " << (output_cpu == output_blelloch ? "✅" : "❌") << "\n";
}

int main() {
    std::vector<int> sizes = {1 << 10, 1 << 14, 1 << 17, 1 << 20, 1 << 23}; // 1K to 8M

    for (int n : sizes) {
        benchmark(n);
    }

    return 0;
}
