#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

__global__ void blelloch_scan_kernel(int* d_out, const int* d_in, int n) {
    extern __shared__ int temp[]; // shared memory
    int tid = threadIdx.x;int offset = 1;int ai = tid;int bi = tid + (n / 2);
    // Load input into shared memory
    if (2 * tid < n) temp[2 * tid] = d_in[2 * tid];
    else temp[2 * tid] = 0;
    if (2 * tid + 1 < n) temp[2 * tid + 1] = d_in[2 * tid + 1];
    else temp[2 * tid + 1] = 0;
    // Up-sweep (reduction)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }
    // Set last element to zero for exclusive scan
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
    // Write results to output
    if (2 * tid < n) d_out[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) d_out[2 * tid + 1] = temp[2 * tid + 1];
}

void blelloch_scan(const std::vector<int>& input, std::vector<int>& output) {
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

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Blelloch GPU Time: " << ms << " ms\n";

    CHECK_CUDA(cudaMemcpy(output.data(), d_out, nextPow2 * sizeof(int), cudaMemcpyDeviceToHost));
    output.resize(n); // truncate to original size

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
