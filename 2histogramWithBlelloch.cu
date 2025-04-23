#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " — " \
                  << cudaGetErrorString(call) << std::endl; \
        exit(1); \
    }

// const int BIN_COUNT = 16;
const int BIN_COUNT = 1024;

// Host function: generate input data
void generateInput(std::vector<int>& data, int numElements, int maxValue = 255) {
    data.resize(numElements);
    for (int& x : data) {
        x = rand() % (maxValue + 1);
    }
}

// CPU histogram
void cpuHistogram(const std::vector<int>& input, std::vector<int>& bins, int binCount) {
    bins.assign(binCount, 0);
    for (int val : input) {
        int bin = (val * binCount) / 256;
        if (bin >= binCount) bin = binCount - 1;
        bins[bin]++;
    }
}

// CPU prefix sum
void cpuScan(const std::vector<int>& input, std::vector<int>& output) {
    output.resize(input.size());
    output[0] = 0;
    for (size_t i = 1; i < input.size(); ++i)
        output[i] = output[i - 1] + input[i - 1];
}

// CUDA kernel for histogram
__global__ void histogram_kernel(int* input, int* histo, int size, int binCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    int bin = (input[idx] * binCount) / 256;
    if (bin >= binCount) bin = binCount - 1;

    atomicAdd(&histo[bin], 1);
}

// Naive scan kernel (not efficient, but illustrative)
__global__ void naive_scan_kernel(int* d_out, const int* d_in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int sum = 0;
    for (int i = 0; i < tid; ++i)
        sum += d_in[i];
    d_out[tid] = sum;
}

// Blelloch scan kernel (shared memory, assumes power of two)
__global__ void blelloch_scan_kernel(int* d_out, const int* d_in, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    int ai = tid;
    int bi = tid + (n / 2);

    temp[2 * tid]     = (2 * tid < n)     ? d_in[2 * tid]     : 0;
    temp[2 * tid + 1] = (2 * tid + 1 < n) ? d_in[2 * tid + 1] : 0;

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

// GPU histogram + scan (common path)
void gpuHistogramWithScan(const std::vector<int>& input, std::vector<int>& prefixOut, bool useBlelloch, float& timeMs) {
    int* d_input = nullptr;
    int* d_hist = nullptr;
    int* d_scan = nullptr;
    int size = input.size();

    CHECK_CUDA(cudaMalloc(&d_input, size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_hist, BIN_COUNT * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int)));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Histogram
    histogram_kernel<<<blocks, threads>>>(d_input, d_hist, size, BIN_COUNT);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMalloc(&d_scan, BIN_COUNT * sizeof(int)));

    // Time prefix sum only
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (useBlelloch) {
        blelloch_scan_kernel<<<1, BIN_COUNT / 2, BIN_COUNT * sizeof(int)>>>(d_scan, d_hist, BIN_COUNT);
    } else {
        naive_scan_kernel<<<1, BIN_COUNT>>>(d_scan, d_hist, BIN_COUNT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeMs, start, stop);

    prefixOut.resize(BIN_COUNT);
    CHECK_CUDA(cudaMemcpy(prefixOut.data(), d_scan, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaFree(d_scan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int inputSize = 1 << 20; // 1 million elements
    std::vector<int> input;
    generateInput(input, inputSize);

    std::vector<int> cpu_hist, cpu_prefix;
    cpuHistogram(input, cpu_hist, BIN_COUNT);
    cpuScan(cpu_hist, cpu_prefix);

    std::vector<int> naive_result, blelloch_result;
    float t_naive = 0.0f, t_blelloch = 0.0f;

    gpuHistogramWithScan(input, naive_result, false, t_naive);
    gpuHistogramWithScan(input, blelloch_result, true, t_blelloch);

    std::cout << "--- Histogram Prefix Sum Benchmark (Input size = " << inputSize << ") ---\n";
    std::cout << "Naive Scan Time:    " << t_naive << " ms, Match? " << (naive_result == cpu_prefix ? "✅" : "❌") << "\n";
    std::cout << "Blelloch Scan Time: " << t_blelloch << " ms, Match? " << (blelloch_result == cpu_prefix ? "✅" : "❌") << "\n";

    return 0;
}
