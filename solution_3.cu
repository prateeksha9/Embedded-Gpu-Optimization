// 2D-convolutional.cu
// Problem 3: Tiled 2-D Convolution Layer (B=1, C=4, H=W=256, M=16, K=3)
// - Shared-memory tiling with halo
// - Zero padding
// - Runtime stride and padding
// - Top function: ConvolutionLayer
// - Main: random data, CPU golden reference, compare

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Fixed layer dimensions per problem statement
#define B 1         // batch
#define C 4         // input channels
#define H 256       // input height
#define W 256       // input width
#define M 16        // output feature maps
#define K 3         // kernel size (K x K)
#define TILE 16     // tile size
#define STRIDE 1    // stride (compile-time constant for static shared memory)

// ---------------------------------------------
// CPU Golden Reference (zero padding, stride S)
// X: [B][C][H][W], Wt: [M][C][K][K], Y: [B][M][Hout][Wout]
// ---------------------------------------------
void ConvolutionLayerCPURef(const float* X, const float* Wt, float* Y,
                       int B_, int C_, int H_, int W_,
                       int M_, int K_, int stride, int pad)
{
    const int Hout = (H_ + 2 * pad - K_) / stride + 1;
    const int Wout = (W_ + 2 * pad - K_) / stride + 1;

    auto Xidx = [B_, C_, H_, W_](int b, int c, int h, int w) {
        return ((b * C_ + c) * H_ + h) * W_ + w;
    };
    auto Widx = [C_, K_](int m, int c, int p, int q) {
        return ((m * C_ + c) * K_ + p) * K_ + q;
    };
    auto Yidx = [B_, M_, Hout, Wout](int b, int m, int h, int w) {
        return ((b * M_ + m) * Hout + h) * Wout + w;
    };

    for (int b = 0; b < B_; ++b) {
        for (int m = 0; m < M_; ++m) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float acc = 0.0f;
                    const int in_h0 = oh * stride - pad;
                    const int in_w0 = ow * stride - pad;
                    for (int c = 0; c < C_; ++c) {
                        for (int p = 0; p < K_; ++p) {
                            for (int q = 0; q < K_; ++q) {
                                const int ih = in_h0 + p;
                                const int iw = in_w0 + q;
                                float x = 0.0f;
                                if (ih >= 0 && ih < H_ && iw >= 0 && iw < W_) {
                                    x = X[Xidx(b, c, ih, iw)];
                                }
                                acc += x * Wt[Widx(m, c, p, q)];
                            }
                        }
                    }
                    Y[Yidx(b, m, oh, ow)] = acc;
                }
            }
        }
    }
}

// -----------------------------------------------------
// CUDA Kernel: Tiled 2D Convolution with shared memory
// - One block computes a TILE x TILE patch of outputs
// - Shared tile size = (TILE*stride + K - 1)
// - Per-channel streaming into shared memory (saves SMEM)
// -----------------------------------------------------
__global__ void conv2d_tiled_kernel(const float* __restrict__ X,   // [B][C][H][W]
                                    const float* __restrict__ Wt,  // [M][C][K][K]
                                    float* __restrict__ Y,         // [B][M][Hout][Wout]
                                    int B_, int C_, int H_, int W_,
                                    int M_, int K_,
                                    int pad,
                                    int Hout, int Wout)
{
    // Get batch index from blockIdx.z
    const int b = blockIdx.z;
    
    // Block's top-left output coordinate
    const int out_h0 = blockIdx.y * TILE;
    const int out_w0 = blockIdx.x * TILE;

    // This thread's output coordinate
    const int oh = out_h0 + threadIdx.y;
    const int ow = out_w0 + threadIdx.x;

    // Shared tile spatial size covering all outputs in this block (compile-time constant)
    constexpr int SH = TILE * STRIDE + (K - 1);  // rows
    constexpr int SW = TILE * STRIDE + (K - 1);  // cols

    // Shared memory size: (TILE*stride + K - 1)^2
    // In this case, (16*1 + 3 - 1)^2 = 18^2
    __shared__ float smem[SH * SW];

    // Accumulators per output feature map
    float acc[M];  // M==16 here
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    // Top-left input coordinate that aligns with block's top-left output
    const int in_h0 = out_h0 * STRIDE - pad;
    const int in_w0 = out_w0 * STRIDE - pad;

    // Lambda to index flattened arrays (including batch dimension)
    auto Xidx = [B_, C_, H_, W_](int b, int c, int h, int w) {
        return ((b * C_ + c) * H_ + h) * W_ + w;
    };
    auto Widx = [C_, K_](int m, int c, int p, int q) {
        return ((m * C_ + c) * K_ + p) * K_ + q;
    };
    auto Yidx = [B_, M_, Hout, Wout](int b, int m, int h, int w) {
        return ((b * M_ + m) * Hout + h) * Wout + w;
    };

    // Stream over channels: load that channel's patch into shared memory, convolve, repeat
    for (int c = 0; c < C_; ++c) {

        // Cooperative load of SH x SW tile for channel c
        for (int r = threadIdx.y; r < SH; r += blockDim.y) {
            const int ih = in_h0 + r;
            for (int col = threadIdx.x; col < SW; col += blockDim.x) {
                const int iw = in_w0 + col;
                float x = 0.0f;
                if (ih >= 0 && ih < H_ && iw >= 0 && iw < W_) {
                    x = X[Xidx(b, c, ih, iw)];
                }
                smem[r * SW + col] = x;  // zero padding automatically when out-of-range
            }
        }
        __syncthreads();

        // Each thread computes contributions for its output (if in bounds)
        if (oh < Hout && ow < Wout) {
            // Location of this output's receptive field top-left inside shared tile
            const int sr = threadIdx.y * STRIDE;
            const int sc = threadIdx.x * STRIDE;

            // For each output feature m: accumulate KxK dot with this channel
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                float sum = 0.0f;
                #pragma unroll
                for (int p = 0; p < K; ++p) {
                    #pragma unroll
                    for (int q = 0; q < K; ++q) {
                        sum += smem[(sr + p) * SW + (sc + q)] * Wt[Widx(m, c, p, q)];
                    }
                }
                acc[m] += sum;
            }
        }

        __syncthreads();
    }

    // Write results
    if (oh < Hout && ow < Wout) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            Y[Yidx(b, m, oh, ow)] = acc[m];
        }
    }
}

// -----------------------------------------------------
// Top function: ConvolutionLayer
// - Allocates device memory (or uses provided device pointers)
// - Launches tiled kernel with given stride & padding
// - Copies result back to host
// -----------------------------------------------------
void ConvolutionLayer(const float* h_X,      // host input  [B][C][H][W]
                      const float* h_Wt,     // host filter [M][C][K][K]
                      float* h_Y,            // host output [B][M][Hout][Wout]
                      int pad)
{
    const int Hout = (H + 2 * pad - K) / STRIDE + 1;
    const int Wout = (W + 2 * pad - K) / STRIDE + 1;

    const size_t bytes_X  = B * C * H * W * sizeof(float);
    const size_t bytes_Wt = M * C * K * K * sizeof(float);
    const size_t bytes_Y  = B * M * Hout * Wout * sizeof(float);

    // Device buffers
    float *d_X = nullptr, *d_Wt = nullptr, *d_Y = nullptr;
    cudaMalloc(&d_X,  bytes_X);
    cudaMalloc(&d_Wt, bytes_Wt);
    cudaMalloc(&d_Y,  bytes_Y);

    cudaMemcpy(d_X,  h_X,  bytes_X,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wt, h_Wt, bytes_Wt, cudaMemcpyHostToDevice);

    // Grid/block config: use blockIdx.z for batch dimension
    dim3 block(TILE, TILE, 1);
    dim3 grid( (Wout + TILE - 1) / TILE, // == ceil(Wout / TILE)
               (Hout + TILE - 1) / TILE, // == ceil(Hout / TILE)
               B);  // B blocks in z-dimension for batch

    // Create timer
    float total_time = 0;
    struct timeval start, end;
    
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    // Launch kernel (no shared memory parameter needed for static)
    conv2d_tiled_kernel<<<grid, block>>>(d_X, d_Wt, d_Y,
                                         B, C, H, W,
                                         M, K,
                                         pad,
                                         Hout, Wout);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
    total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    
    // Print kernel latency
    printf("latency: %.3f ms\n", total_time);

    // Copy back
    cudaMemcpy(h_Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Wt);
    cudaFree(d_Y);
}

// -----------------------------------------------------
// Evaluation (main)
// - Generate random input and weights
// - CPU golden reference
// - GPU result via ConvolutionLayer
// - Compare
// -----------------------------------------------------
int main() {
    // Runtime parameters (change as needed)
    const int pad = K / 2;      // zero padding for "same" when STRIDE==1

    const int Hout = (H + 2 * pad - K) / STRIDE + 1;
    const int Wout = (W + 2 * pad - K) / STRIDE + 1;

    // Allocate host buffers (including batch dimension)
    std::vector<float> h_X(B * C * H * W);
    std::vector<float> h_Wt(M * C * K * K);
    std::vector<float> h_Y_gpu(B * M * Hout * Wout, 0.0f);
    std::vector<float> h_Y_cpu(B * M * Hout * Wout, 0.0f);

    // Random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_X)  v = dist(rng);
    for (auto& w : h_Wt) w = dist(rng);

    // Create timer for CPU
    float total_time_CPU = 0;
    struct timeval start_CPU, end_CPU;
    gettimeofday(&start_CPU, NULL);
    
    // CPU golden
    ConvolutionLayerCPURef(h_X.data(), h_Wt.data(), h_Y_cpu.data(),
                      B, C, H, W, M, K, STRIDE, pad);
    
    gettimeofday(&end_CPU, NULL);
    total_time_CPU += (end_CPU.tv_sec - start_CPU.tv_sec) * 1000 + (end_CPU.tv_usec - start_CPU.tv_usec) * 0.001;
    //print CPU latency
    printf("CPU latency: %.3f ms\n",  total_time_CPU);

    // GPU
    ConvolutionLayer(h_X.data(), h_Wt.data(), h_Y_gpu.data(), pad);

    //-----------------------------------------------------
    // Validate
    //-----------------------------------------------------

    int error_count = 0;

    printf("Comparison (B=%d, STRIDE=%d, pad=%d): Hout=%d, Wout=%d\n", B, STRIDE, pad, Hout, Wout);
    for (size_t i = 0; i < h_Y_cpu.size(); ++i) {
        double diff = static_cast<double>(h_Y_cpu[i]) - static_cast<double>(h_Y_gpu[i]);
        if (diff > 1e-4) {
            printf("Failed at index %zu: CPU: %.6g, GPU: %.6g\n", i, h_Y_cpu[i], h_Y_gpu[i]);
            error_count++;
        }
    }
    
    if (error_count > 0) {
        printf("FAIL ❌\n");
    } else {
        printf("PASS ✅\n");
    }

    return 0;
}

/*
Build:
  nvcc -O2 -arch=sm_86 2D-convolutional.cu -o conv2d_tiled

Run:
  ./conv2d_tiled

Notes:
- TILE=16 works well on most GPUs. You can -DTILE=8 or 32 at compile time to experiment:
    nvcc -O2 -arch=sm_86 -DTILE=32 2D-convolutional.cu -o conv2d_tiled
- Shared memory per block:
    SH = TILE*stride + (K-1)
    smem_bytes = (SH*SH*sizeof(float))
  For stride=1, K=3, TILE=16  → SH=18 → ~1.3 KB per block.
  For stride=2, K=3, TILE=16  → SH=34 → ~4.6 KB per block.
- We stream channels into shared memory (one channel at a time) to keep SMEM small,
  and keep per-output accumulators in registers (acc[16]).
- Zero padding is handled during the shared-memory loads by bounds checks.
*/
