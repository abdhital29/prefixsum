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

// Step 2: Blelloch Scan (shared memory)
__global__ void blelloch_scan(int* output, const int* input, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    int ai = 2 * tid;
    int bi = 2 * tid + 1;

    // Load input into shared memory
    temp[ai] = (ai < n) ? input[ai] : 0;
    temp[bi] = (bi < n) ? input[bi] : 0;

    // Up-sweep (reduce) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            temp[j] += temp[i];
        }
        offset <<= 1;
    }

    // Clear the last element for exclusive scan
    if (tid == 0) temp[n - 1] = 0;

    // Down-sweep phase
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            int t = temp[i];
            temp[i] = temp[j];
            temp[j] += t;
        }
    }

    __syncthreads();
    if (ai < n) output[ai] = temp[ai];
    if (bi < n) output[bi] = temp[bi];
}

// Step 3: Scatter
__global__ void scatter(const int* input, const int* flags, const int* scanned, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i] == 1)
        output[scanned[i]] = input[i];
}

// Host function with Blelloch scan
void stream_compaction_blelloch(const std::vector<int>& input, std::vector<int>& compacted, int& valid_count) {
    int n = input.size();
    int nextPow2 = 1;
    while (nextPow2 < n) nextPow2 <<= 1;

    int* d_input = nullptr;
    int* d_flags = nullptr;
    int* d_scanned = nullptr;
    int* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_flags, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_scanned, nextPow2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int threads = nextPow2 / 2;
    dim3 blocks((n + 255) / 256);
    dim3 threads_per_block(256);

    // Step 1: Mark flags
    mark_flags<<<blocks, threads_per_block>>>(d_input, d_flags, n);

    // Step 2: Blelloch scan
    blelloch_scan<<<1, threads, nextPow2 * sizeof(int)>>>(d_scanned, d_flags, nextPow2);

    // Step 3: Scatter
    scatter<<<blocks, threads_per_block>>>(d_input, d_flags, d_scanned, d_output, n);

    // Retrieve final count
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
