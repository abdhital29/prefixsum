#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

// CPU Prefix Sum (Exclusive)
void cpu_scan(const std::vector<int>& input, std::vector<int>& output) {
    output[0] = 0;
    for (size_t i = 1; i < input.size(); ++i)
        output[i] = output[i - 1] + input[i - 1];
}

// Naive GPU Kernel
__global__ void naive_scan_kernel(int* d_out, const int* d_in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int sum = 0;
    for (int i = 0; i < tid; ++i)
        sum += d_in[i];
    d_out[tid] = sum;
}

void naive_gpu_scan(const std::vector<int>& input, std::vector<int>& output, float& time_ms) {
    int* d_in = nullptr;
    int* d_out = nullptr;
    int n = input.size();
    output.resize(n);

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

    cudaEventElapsedTime(&time_ms, start, stop);
    CHECK_CUDA(cudaMemcpy(output.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Blelloch GPU Kernel
__global__ void blelloch_scan_kernel(int* d_out, const int* d_in, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    int ai = tid;
    int bi = tid + (n / 2);

    // Load input into shared memory
    if (2 * tid < n) temp[2 * tid] = d_in[2 * tid];
    else temp[2 * tid] = 0;

    if (2 * tid + 1 < n) temp[2 * tid + 1] = d_in[2 * tid + 1];
    else temp[2 * tid + 1] = 0;

    // Up-sweep
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Clear last element for exclusive scan
    if (tid == 0) temp[n - 1] = 0;

    // Down-sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    if (2 * tid < n) d_out[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) d_out[2 * tid + 1] = temp[2 * tid + 1];
}

void blelloch_gpu_scan(const std::vector<int>& input, std::vector<int>& output, float& time_ms) {
    int n = input.size();
    int nextPow2 = 1;
    while (nextPow2 < n) nextPow2 <<= 1;

    std::vector<int> padded_input = input;
    padded_input.resize(nextPow2, 0);
    output.resize(nextPow2);

    int* d_in = nullptr;
    int* d_out = nullptr;

    CHECK_CUDA(cudaMalloc(&d_in, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_in, padded_input.data(), nextPow2 * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockDim(nextPow2 / 2);
    dim3 gridDim(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blelloch_scan_kernel<<<gridDim, blockDim, nextPow2 * sizeof(int)>>>(d_out, d_in, nextPow2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    CHECK_CUDA(cudaMemcpy(output.data(), d_out, nextPow2 * sizeof(int), cudaMemcpyDeviceToHost));
    output.resize(n);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

double max_absolute_error(const std::vector<int>& ref, const std::vector<int>& test) {
    double max_err = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double err = std::abs(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

void benchmark(int n) {
    std::vector<int> input(n);
    for (int& x : input) x = rand() % 10;

    std::vector<int> out_cpu(n), out_naive(n), out_blelloch(n);
    float t_naive = 0.0f, t_blelloch = 0.0f;

    // CPU
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_scan(input, out_cpu);
    auto t2 = std::chrono::high_resolution_clock::now();
    double t_cpu = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Naive GPU
    naive_gpu_scan(input, out_naive, t_naive);

    // Blelloch GPU
    blelloch_gpu_scan(input, out_blelloch, t_blelloch);

    // Error checks
    double err_naive = max_absolute_error(out_cpu, out_naive);
    double err_blelloch = max_absolute_error(out_cpu, out_blelloch);

    std::cout << "\n--- Benchmark n = " << n << " ---\n";
    std::cout << "CPU:      " << t_cpu     << " ms\n";
    std::cout << "Naive GPU:" << t_naive   << " ms  Match? " << (out_naive == out_cpu ? "✅" : "❌")
              << "  Max Error: " << err_naive << "\n";
    std::cout << "Blelloch: " << t_blelloch << " ms  Match? " << (out_blelloch == out_cpu ? "✅" : "❌")
              << "  Max Error: " << err_blelloch << "\n";
}

int main() {
    std::vector<int> sizes = {1 << 10, 1 << 14, 1 << 18, 1 << 22}; // 1K, 16K, 256K, 4M
    for (int n : sizes) {
        benchmark(n);
    }
    return 0;
}
