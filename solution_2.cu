// conv1d_tiled.cu
// Tiled 1-D convolution with shared-memory halos and zero padding.
// N = 2048, MASK_WIDTH = 5 (radius R=2)

#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define N            2048
#define MASK_WIDTH   5
#define R            (MASK_WIDTH / 2) // radius of the mask for halo regions
#define BLOCK_SIZE   16

// Put the mask in constant memory (fast, cached)
__constant__ float d_mask[MASK_WIDTH];

// -----------------------------
// Device kernel (tiled + halos)
// -----------------------------
/**
 * @brief CUDA kernel for 1D convolution with tiled shared memory and halos.
 * 
 * Performs 1D convolution using shared memory tiling with left/right halo regions
 * for efficient memory access. Uses zero padding for boundary conditions.
 * 
 * @param d_in Input array in device memory (read-only, __restrict__ ensures no aliasing)
 * @param d_out Output array in device memory (write-only, __restrict__ ensures no aliasing)
 * @param n Size of the input/output arrays
 * 
 * @note __restrict__ qualifier ensures no aliasing between d_in and d_out, enabling
 *       compiler optimizations. __global__ qualifier makes this a CUDA kernel that
 *       can be launched from the host and executed on the device.
 */
__global__ void BasicConvolution(const float* __restrict__ d_in,
                                  float* __restrict__ d_out,
                                  int n)
{
    // Shared memory: tile + left/right halos
    __shared__ float tile[BLOCK_SIZE + 2 * R];

    const int tid     = threadIdx.x;
    const int g_start = blockIdx.x * BLOCK_SIZE;      // first global index this block covers
    const int g_idx   = g_start + tid;                // global index for this thread
    const int l_idx   = tid + R;                      // local index in shared (after left halo)

    // ---- Load main tile (coalesced) ----
    tile[l_idx] = (g_idx < n) ? d_in[g_idx] : 0.0f;

    // ---- Load halos (first R threads per side) ----
    if (tid < R) {
        // Left halo element for this tid
        int left_g = g_start + tid - R; // may be < 0
        tile[tid] = (left_g >= 0) ? d_in[left_g] : 0.0f;  // zero padding

        // Right halo element for this tid
        int right_g = g_start + BLOCK_SIZE + tid; // may be >= n
        tile[l_idx + BLOCK_SIZE] = (right_g < n) ? d_in[right_g] : 0.0f; // zero padding
    }

    __syncthreads();

    // ---- Compute convolution (only if in range) ----
    if (g_idx < n) {
        float acc = 0.0f;
        #pragma unroll
        // index range: [l_idx - R, l_idx + R]
        for (int k = -R; k <= R; ++k) {
            acc += d_mask[k + R] * tile[l_idx + k];
        }
        d_out[g_idx] = acc;
    }
}

// -----------------------------
// CPU reference (zero padding)
// -----------------------------
void BasicConvolutionCPURef(const std::vector<float>& in,
                    const std::vector<float>& mask,
                    std::vector<float>& out)
{
    const int n = static_cast<int>(in.size());
    out.assign(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int k = -R; k <= R; ++k) {
            int j = i + k;
            float x = (j >= 0 && j < n) ? in[j] : 0.0f; // zero padding
            acc += mask[k + R] * x;
        }
        out[i] = acc;
    }
}

int main()
{
    // -----------------------------
    // Host data
    // -----------------------------
    std::vector<float> h_in(N), h_out(N), h_ref(N), h_mask(MASK_WIDTH);

    // Initialize random seed
    srand(time(0));

    // Generate random input data
    for (int i = 0; i < N; ++i) h_in[i] = rand() / (float)RAND_MAX;

    // Simple symmetric mask (sum to ~1)
    h_mask[0] = 0.0625f; h_mask[1] = 0.25f; h_mask[2] = 0.375f; h_mask[3] = 0.25f; h_mask[4] = 0.0625f;


    // Create timer for CPU
    float total_time_CPU = 0;
    struct timeval start_CPU, end_CPU;
    gettimeofday(&start_CPU, NULL);
    
    // Reference on CPU
    BasicConvolutionCPURef(h_in, h_mask, h_ref);
    
    gettimeofday(&end_CPU, NULL);
    total_time_CPU += (end_CPU.tv_sec - start_CPU.tv_sec) * 1000 + (end_CPU.tv_usec - start_CPU.tv_usec) * 0.001;
    //print CPU latency
    printf("CPU latency: %.3f ms\n",  total_time_CPU);

    // -----------------------------
    // Device alloc + copy
    // -----------------------------
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask.data(), MASK_WIDTH * sizeof(float));

    // -----------------------------
    // Launch kernel
    // -----------------------------
    const dim3 block(BLOCK_SIZE); // number of threads per block
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE); // number of blocks needed to cover the input array

    // Create timer for GPU
    float total_time = 0;
    struct timeval start, end;
    
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    // Call the kernel
    BasicConvolution<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
    total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    
    // Print kernel latency
    printf("latency: %.3f ms\n", total_time);

    // Copy back
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // -----------------------------
    // Validate
    // -----------------------------
    const double tolerance = 1e-5;  // Maximum allowed error
    double max_abs_err = 0.0;
    for (int i = 0; i < N; ++i) {
        max_abs_err = std::max(max_abs_err, static_cast<double>(std::fabs(h_out[i] - h_ref[i])));
    }
    
    if (max_abs_err > tolerance) {
        fprintf(stderr, "ERROR: Validation failed! Max absolute error (%.6g) exceeds tolerance (%.6g)\n", 
                max_abs_err, tolerance);
    } else {
        printf("Validation passed!\n");
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

/*
Build:
  nvcc -O3 -arch=sm_86 -use_fast_math --ptxas-options=-v 1D-convolutional.cu -o 1d_conv
  
  -O3: Maximum optimization level
  -arch=sm_86: Target compute capability 8.6 (adjust for your GPU)
  -use_fast_math: Enable fast floating-point math (may sacrifice precision)
  --ptxas-options=-v: Verbose PTX assembler output (shows register usage, shared memory)

Run:
  ./1d_conv

Notes:
- Zero padding is handled both in the shared-memory halo loads (left_g<0 or right_g>=N)
  and by guarding main loads (g_idx>=N).
- Change BLOCK_SIZE to 256 for higher occupancy if shared memory allows on your GPU.
- The mask is in __constant__ memory for fast broadcast to threads.
*/
