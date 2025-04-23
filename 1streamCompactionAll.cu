// streamCompactionBenchmark.cu
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

__global__ void mapToBooleanKernel(int* d_flags, const int* d_input, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) d_flags[i] = (d_input[i] != 0) ? 1 : 0;
}

__global__ void scatterKernel(int* d_output, const int* d_input, const int* d_flags, const int* d_indices, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n && d_flags[i]) d_output[d_indices[i]] = d_input[i];
}

__global__ void naiveScanKernel(int* d_out, const int* d_in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int sum = 0;
    for (int i = 0; i < tid; ++i)
        sum += d_in[i];
    d_out[tid] = sum;
}

__global__ void blellochScanKernel(int* d_out, const int* d_in, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    int ai = tid;
    int bi = tid + (n / 2);

    if (2 * tid < n) temp[2 * tid] = d_in[2 * tid];
    else temp[2 * tid] = 0;

    if (2 * tid + 1 < n) temp[2 * tid + 1] = d_in[2 * tid + 1];
    else temp[2 * tid + 1] = 0;

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (tid == 0) temp[n - 1] = 0;

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

void streamCompactionCPU(const std::vector<int>& input, std::vector<int>& output, float& time_ms) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int x : input) {
        if (x != 0) output.push_back(x);
    }

    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<float, std::milli>(end - start).count();
}

void streamCompactionNaive(const std::vector<int>& input, std::vector<int>& output, float& time_ms) {
    int n = input.size();
    int* d_input = nullptr, * d_flags = nullptr, * d_indices = nullptr, * d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_flags, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_indices, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mapToBooleanKernel<<<blocks, threads>>>(d_flags, d_input, n);
    naiveScanKernel<<<blocks, threads>>>(d_indices, d_flags, n);
    scatterKernel<<<blocks, threads>>>(d_output, d_input, d_flags, d_indices, n);

    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventElapsedTime(&time_ms, start, stop);

    std::vector<int> indices(n);
    CHECK_CUDA(cudaMemcpy(indices.data(), d_indices, n * sizeof(int), cudaMemcpyDeviceToHost));
    int new_size = indices.back();
    if (input.back() != 0) new_size++;

    output.resize(new_size);
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, new_size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_flags);
    cudaFree(d_indices);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void streamCompactionBlelloch(const std::vector<int>& input, std::vector<int>& output, float& time_ms) {
    int n = input.size();
    int nextPow2 = 1;
    while (nextPow2 < n) nextPow2 <<= 1;

    std::vector<int> padded_input = input;
    padded_input.resize(nextPow2, 0);

    int* d_input = nullptr, * d_flags = nullptr, * d_indices = nullptr, * d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_flags, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_indices, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input, padded_input.data(), nextPow2 * sizeof(int), cudaMemcpyHostToDevice));

    int threads = nextPow2 / 2;
    int blocks = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mapToBooleanKernel<<<(nextPow2 + 255) / 256, 256>>>(d_flags, d_input, nextPow2);
    blellochScanKernel<<<blocks, threads, nextPow2 * sizeof(int)>>>(d_indices, d_flags, nextPow2);
    scatterKernel<<<(nextPow2 + 255) / 256, 256>>>(d_output, d_input, d_flags, d_indices, nextPow2);

    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventElapsedTime(&time_ms, start, stop);

    std::vector<int> indices(nextPow2);
    CHECK_CUDA(cudaMemcpy(indices.data(), d_indices, nextPow2 * sizeof(int), cudaMemcpyDeviceToHost));
    int new_size = indices[n - 1];
    if (input[n - 1] != 0) new_size++;

    output.resize(new_size);
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, new_size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_flags);
    cudaFree(d_indices);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = 1 << 20; // 1M elements
    std::vector<int> input(N);
    for (int& x : input) x = (rand() % 5 == 0) ? 0 : (rand() % 100);

    std::vector<int> out_cpu, out_naive, out_blelloch;
    float t_cpu = 0.0f, t_naive = 0.0f, t_blelloch = 0.0f;

    streamCompactionCPU(input, out_cpu, t_cpu);
    streamCompactionNaive(input, out_naive, t_naive);
    streamCompactionBlelloch(input, out_blelloch, t_blelloch);

    std::cout << "--- Stream Compaction Benchmark (Input size = " << N << ") ---\n";
    std::cout << "CPU Time:      " << t_cpu << " ms, Output size: " << out_cpu.size() << "\n";
    std::cout << "Naive Time:    " << t_naive << " ms, Output size: " << out_naive.size()
              << ", Match CPU? " << ((out_cpu == out_naive) ? "✅" : "❌") << "\n";
    std::cout << "Blelloch Time: " << t_blelloch << " ms, Output size: " << out_blelloch.size()
              << ", Match CPU? " << ((out_cpu == out_blelloch) ? "✅" : "❌") << "\n";

    return 0;
}
