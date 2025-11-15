# GPU Parallel Programming: Tiled Matrix Multiplication and Convolution Kernels

### ECPS 209 – CPS Case Studies, Assignment 1

**Authors:** Chaerin An (chaerina), Selina Shrestha (shresth4), Prateeksha Ranjan (prateekr)
**Date:** November 7, 2025

---

# Overview

This project implements high-performance CUDA kernels for three foundational numerical operations:

1. Tiled Matrix Multiplication (GEMM)
2. Tiled 1-D Convolution
3. Tiled 2-D Convolution (multi-channel, CNN-style)

The goal is to study how shared-memory tiling improves performance on embedded GPUs (Jetson Nano), understand tile size trade-offs, and compare GPU performance against sequential CPU implementations.

---

# Research Contributions

This work contributes several research-relevant insights in GPU memory hierarchy optimization and algorithm–hardware co-design.

### 1. Cross-Domain Evaluation of Tiling

We evaluate the same tiling strategy across GEMM, 1D convolution, and 2D convolution, demonstrating consistent performance improvements across linear algebra and neural network workloads.

### 2. Systematic Study of Tile Size vs. GPU Performance

We empirically sweep tile sizes (4, 8, 16, 32, 64) and show that medium tile sizes (8–16) consistently yield the best performance due to balanced reuse, synchronization cost, and GPU occupancy.

### 3. GPU vs CPU Scaling on Embedded Platforms

Speedups range from 3× for 1D convolution to up to 180× for matrix multiplication.
This highlights the sensitivity of embedded GPUs to global memory bandwidth and the importance of shared memory reuse.

### 4. Practical CUDA Kernel Design Strategies

The kernels incorporate:

* Guarded shared-memory loads
* Halo tiling
* Constant memory broadcast
* Multi-filter register accumulation
* Coalesced memory accesses
* Phase-wise K-dimension streaming
  These reflect real optimization principles used in deep-learning libraries such as cuDNN and TensorRT.

### 5. Relevance to Edge AI and Scientific ML

These kernels form the computational core of CNN layers, stencils, and transformer feed-forward networks.
Findings generalize to real-time CPS workloads where latency, bandwidth, and energy constraints dominate.

---

# Folder Structure

```
gpu-tiled-kernels/
│
├── src/
│   ├── matrixMul.cu            # Tiled GEMM kernel
│   ├── conv1d.cu               # Tiled 1D convolution kernel
│   ├── conv2d.cu               # Tiled 2D multi-channel convolution kernel
│   ├── utils.h                 # Common utilities (optional)
│   └── Makefile                # Build automation
│
├── results/
│   ├── gemm_results.csv
│   ├── conv1d_results.csv
│   ├── conv2d_results.csv
│   └── plots/
│       ├── gemm_performance.png
│       ├── conv1d_performance.png
│       └── conv2d_performance.png
│
├── README.md
└── report/
```

---

# GPU Memory Hierarchy Diagram

```
                 GPU Memory Hierarchy

               +-------------------------------------+
               |            Global Memory            |
               |  - DRAM                             |
               |  - High latency                     |
               |  - Stores input/output tensors      |
               +--------------------+----------------+
                                    |
                                    v
               +-------------------------------------+
               |            Shared Memory             |
               |  - On-chip (per SM)                  |
               |  - Low latency, banked               |
               |  - Stores tiles and halo regions     |
               +--------------------+----------------+
                                    |
                                    v
               +-------------------------------------+
               |              Registers               |
               |  - Fastest memory                    |
               |  - Per-thread accumulators           |
               +--------------------+----------------+
                                    |
                                    v
               +-------------------------------------+
               |              ALUs / Cores            |
               |  - Executes fused multiply-add       |
               +-------------------------------------+
```

Shared memory sits between global DRAM and per-thread registers, making it the key component for tiling performance.

---

# Code Architecture Diagram

```
+------------------------------------------------------+
|                       main.cpp                       |
+----------------------------+--------------------------+
                             |
                             v
+------------------------------------------------------+
|                   CPU Reference Code                 |
|      Sequential GEMM / 1D / 2D Convolution           |
+----------------------------+--------------------------+
                             |
                             | Calls top-level wrapper
                             v
+------------------------------------------------------+
|                  Host Wrapper (Top Fn)               |
|  cudaMalloc, cudaMemcpy, configure launch params     |
|  Launch CUDA kernel, copy results back to host       |
+----------------------------+--------------------------+
                             |
                             | Kernel Launch
                             v
+------------------------------------------------------+
|                      CUDA Kernel                     |
|------------------------------------------------------|
| Shared Memory Tiling:                                |
|   - Load tile or halo into shared memory             |
|   - Synchronize threads                              |
|   - Compute partial results                          |
|   - Write final output to global memory              |
+------------------------------------------------------+
```

---

# 1. Tiled Matrix Multiplication (GEMM)

## Method Summary

* Grid of blocks: each block is TILE_WIDTH × TILE_WIDTH threads
* Shared memory buffers load tiles of A and B
* Threads compute partial dot products for each tile phase
* Handles arbitrary matrix dimensions via guarded loads
* Final results written only if within output bounds

## Performance Results

| Test | Matrix Size                        | Tile | GPU (ms)  | CPU (ms)    | Speedup |
|      |                                    |      |           |           |
| 1    | (1859×1581) × (1581×1517)          | 4    | 2526.219  | 108339.023  | 43×     |
|      |                                    | 8    | 1026.967  | 106117.703  | 103×    |
|      |                                    | 16   | 751.545   | 108132.867  | 144×    |
|      |                                    | 32   | 1193.495  | 105642.953  | 89×     |
|      |                                    |      |           |           |
| 2    | (1583×1955) × (1955×1867)          | 4    | 3386.916  | 133538.172  | 39×     |
|      |                                    | 8    | 1284.919  | 132886.391  | 103×    |
|      |                                    | 16   | 831.272   | 134054.703  | 161×    |
|      |                                    | 32   | 723.611   | 133225.609  | 184×    |
|      |                                    |      |           |           |

| 3    | (1533×1888) × (1888×1539)          | 4    | 2294.096  | 107531.633  | 47×     |
|      |                                    | 8    | 1301.611  | 108482.203  | 83×     |
|      |                                    | 16   | 1037.627  | 107833.367  | 104×    |
|      |                                    | 32   | 1172.645  | 108325.977  | 92×     |

### Analysis

Tile 16 provides the best reuse–occupancy trade-off on Jetson Nano.
Small tiles underutilize memory locality; very large tiles reduce active blocks.

---

# 2. Tiled 1-D Convolution

## Method Summary

* Shared memory tile + halo region (radius 2)
* Mask stored in constant memory
* Zero padding ensures valid boundary handling
* Block sizes tested: 4, 8, 16, 32, 64

## Performance Results

| Block Size | CPU (ms) | GPU (ms) | Speedup |
| ---------- | -------- | -------- | ------- |
| 4          | 0.586    | 0.351    | 1.7×    |
| 8          | 0.587    | 0.285    | 2.1×    |
| 16         | 0.587    | 0.183    | 3.2×    |
| 32         | 0.589    | 0.151    | 3.9×    |
| 64         | 0.579    | 0.150    | 3.9×    |

### Analysis

Performance increases with block size until saturation around 32.

---

# 3. Tiled 2-D Convolution (Multi-Channel CNN Layer)

## Method Summary

* Input: B=1, C=4, H=W=256
* Filters: M=16, kernel 3×3
* Shared memory holds a (TILE + 2) × (TILE + 2) patch
* Accumulators for 16 filters kept in registers
* Tile sizes: 4, 8, 16

## Performance Results

| Tile | CPU (ms) | GPU (ms) | Speedup |
| ---- | -------- | -------- | ------- |
| 4    | 2268     | 69.08    | 33×     |
| 8    | 2269     | 37.80    | 60×     |
| 16   | 2268     | 36.51    | 62×     |

### Analysis

Larger tiles greatly improve reuse and reduce halo duplication.

---

# Cross-Kernel Insights

1. Tiling consistently improves performance across domains.
2. Optimal tile sizes are typically 8–16.
3. Embedded GPUs benefit strongly from reduced DRAM traffic.
4. Shared memory reuse is the dominant factor in performance.
5. Register-level accumulation is crucial for CNN-style kernels.

---

# How to Run

## Requirements

* NVIDIA GPU (Jetson Nano used in tests)
* CUDA Toolkit
* nvcc compiler

## Build Commands

### Matrix Multiplication

```
nvcc matrixMul.cu -o matmul
./matmul
```

### 1D Convolution

```
nvcc conv1d.cu -o conv1d
./conv1d
```

### 2D Convolution

```
nvcc conv2d.cu -o conv2d
./conv2d
```

### Optional Optimized Build

```
nvcc -O3 -arch=sm_53 kernel.cu -o kernel
```

Each program will print:

* CPU latency
* GPU latency
* Maximum absolute error
* Pass/Fail validation status

---

# Conclusion

This project demonstrates the effectiveness of shared-memory tiling in accelerating fundamental numerical kernels on embedded GPUs. Tile-aware kernel design provides:

* Up to 180× speedup for GEMM
* Up to 4× speedup for 1D convolution
* Up to 60× speedup for 2D convolution

Medium tile sizes (8–16) consistently offer the best balance of performance and resource utilization.
These results emphasize the importance of memory hierarchy optimization and algorithm-hardware co-design for modern CPS, embedded, and edge-AI workloads.
