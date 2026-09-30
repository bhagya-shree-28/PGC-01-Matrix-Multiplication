# Parallel Matrix Multiplication Using Sequential, OpenMP, MPI and CUDA

## Executive Summary

This repository presents an experimental implementation and performance comparison of matrix multiplication using four different computing approaches: Sequential CPU execution, OpenMP shared-memory parallelism, MPI distributed-memory parallelism, and CUDA GPU parallelism.

The same `4000 × 4000` matrix multiplication workload is implemented using each approach to provide a common basis for comparison. The experiment focuses on understanding how different parallel computing models execute the same computational workload and how factors such as CPU threads, MPI processes, GPU threads, memory architecture, communication, synchronization, and data transfer affect performance.

The experiment includes the complete implementation and execution process for all four approaches, result verification, execution-time measurement, speedup calculation, performance comparison, graph generation, and technical analysis. The Sequential implementation provides the baseline, OpenMP demonstrates multi-threaded CPU execution, MPI demonstrates distributed-memory execution using multiple processes, and CUDA demonstrates GPU-based parallel execution.

The measured execution times are compared to understand the performance characteristics and trade-offs of each computing model for a computationally intensive matrix multiplication workload.

---

## Table of Contents

1. [Experiment Objectives](#1-experiment-objectives)
2. [Theoretical and Architecture Comparison](#2-theoretical-and-architecture-comparison)
   - [2.1 Sequential Computing](#21-sequential-computing)
   - [2.2 OpenMP](#22-openmp)
   - [2.3 MPI](#23-mpi)
   - [2.4 CUDA](#24-cuda)
   - [2.5 Architecture Comparison](#25-architecture-comparison)
3. [Workload Specifications](#3-workload-specifications)
   - [3.1 Matrix Configuration](#31-matrix-configuration)
   - [3.2 Computational Complexity](#32-computational-complexity)
   - [3.3 OpenMP Configuration](#33-openmp-configuration)
   - [3.4 MPI Configuration](#34-mpi-configuration)
   - [3.5 CUDA Configuration](#35-cuda-configuration)
4. [Repository Structure](#4-repository-structure)
5. [Execution Steps](#5-execution-steps)
   - [5.1 Sequential Implementation](#51-sequential-implementation)
   - [5.2 OpenMP Implementation](#52-openmp-implementation)
   - [5.3 MPI Implementation](#53-mpi-implementation)
   - [5.4 CUDA Implementation](#54-cuda-implementation)
6. [Experimental Results](#6-experimental-results)
7. [Performance Comparison](#7-performance-comparison)
   - [7.1 Speedup](#71-speedup)
   - [7.2 Execution Time Comparison](#72-execution-time-comparison)
   - [7.3 Speedup Comparison](#73-speedup-comparison)
8. [Technical Analysis](#8-technical-analysis)
9. [Conclusion](#9-conclusion)

---

# 1. Experiment Objectives

The main objectives of this experiment are:

- To implement matrix multiplication using sequential CPU execution.
- To implement matrix multiplication using OpenMP-based shared-memory parallelism.
- To implement matrix multiplication using MPI-based distributed-memory parallelism.
- To implement matrix multiplication using CUDA-based GPU parallelism.
- To execute the same `4000 × 4000` matrix multiplication workload using all four approaches.
- To measure and record the execution time of each implementation.
- To verify the correctness of the computed matrix results.
- To calculate the speedup of the parallel implementations relative to the sequential baseline.
- To compare the architectural characteristics of Sequential, OpenMP, MPI, and CUDA approaches.
- To analyze the effect of CPU threads, MPI processes, GPU threads, communication, synchronization, and memory transfer on performance.
- To generate graphs for visual comparison of execution time and speedup.
- To understand the practical performance differences between shared-memory, distributed-memory, and GPU-based parallel computing.

---

# 2. Theoretical and Architecture Comparison

## 2.1 Sequential Computing

Sequential computing performs the complete matrix multiplication using a single CPU execution flow.

For two matrices `A` and `B`, the result matrix `C` is calculated as:

```text
C = A × B
```

Each element of the result matrix is calculated using:

```text
C[i][j] = Σ A[i][k] × B[k][j]
```

The computation is performed using three nested loops:

```text
for each row i
    for each column j
        for each k
            C[i][j] += A[i][k] × B[k][j]
```

Since only one execution flow performs the computation, there is no explicit parallelism. This implementation is used as the baseline against which the performance of OpenMP, MPI, and CUDA is compared.

## 2.2 OpenMP

OpenMP is used to introduce shared-memory parallelism on the CPU.

Instead of processing all matrix rows sequentially, the workload is divided among multiple CPU threads. In this experiment, the OpenMP implementation uses:

```text
8 CPU threads
```

The outer loop of the matrix multiplication is parallelized so that different threads can process different rows concurrently.

Conceptually:

```text
                    Shared CPU Memory
                           |
        +------------------+------------------+
        |                  |                  |
     Thread 1           Thread 2          Thread 3
        |                  |                  |
      Rows               Rows               Rows
        |                  |                  |
        +------------------+------------------+
                           |
                       Thread 8
```

All OpenMP threads belong to the same process and share access to the matrices stored in memory.

The main characteristics of OpenMP in this experiment are:

- Shared-memory execution
- Multiple CPU threads
- Same address space
- Low communication overhead
- Parallel execution of independent loop iterations

The OpenMP implementation uses the following directive to distribute loop iterations among threads:

```c
#pragma omp parallel for
```

## 2.3 MPI

MPI (Message Passing Interface) is used to implement distributed-memory parallelism.

Unlike OpenMP, MPI processes do not share the same memory space. Each MPI process has its own address space and communicates with other processes using message-passing operations.

This experiment uses:

```text
4 MPI processes
```

The 4000 rows of the matrix are divided equally among the four processes:

```text
Rank 0 → 1000 rows
Rank 1 → 1000 rows
Rank 2 → 1000 rows
Rank 3 → 1000 rows
```

The general execution flow is:

```text
                    Master Process
                         |
                    MPI_Scatter
                         |
        +----------------+----------------+----------------+
        |                |                |                |
      Rank 0           Rank 1           Rank 2           Rank 3
    1000 rows        1000 rows        1000 rows        1000 rows
        |                |                |                |
        +----------------+----------------+----------------+
                         |
                  Local Computation
                         |
                    MPI_Gather
                         |
                  Complete Matrix C
```

Matrix B is made available to all processes using:

```text
MPI_Bcast
```

The major MPI operations used in the experiment are:

- `MPI_Scatter`
- `MPI_Bcast`
- Local Matrix Multiplication
- `MPI_Gather`

MPI introduces communication and synchronization overhead because data must be distributed between processes and the partial results must be collected after computation.

## 2.4 CUDA

CUDA is used to implement matrix multiplication using GPU parallelism.

Unlike CPU-based Sequential, OpenMP, and MPI implementations, CUDA executes the matrix multiplication using GPU threads.

Each CUDA thread is assigned the computation of an output matrix element.

The experiment uses:

```text
Block size       = 16 × 16
Threads/block    = 256
Grid size        = 250 × 250
Total blocks     = 62,500
```

The execution can be represented as:

```text
                         GPU
                          |
                     CUDA Grid
                          |
        +-----------------+-----------------+
        |                 |                 |
      Block 0           Block 1          Block 2
        |                 |                 |
     16 × 16           16 × 16          16 × 16
     Threads           Threads          Threads
        |                 |                 |
        +-----------------+-----------------+
                          |
                    Result Matrix C
```

Since the matrix has 4000 × 4000 output elements, a large number of GPU threads can work on different output elements concurrently.

The CUDA implementation involves:

1. Allocate host memory
2. Initialize input matrices
3. Allocate device memory
4. Copy matrices from CPU memory to GPU memory
5. Launch CUDA kernel
6. Perform matrix multiplication on GPU
7. Copy result from GPU memory to CPU memory
8. Verify the result

CUDA also introduces host-to-device and device-to-host memory transfers, which contribute to the overall execution time.

## 2.5 Architecture Comparison

| Feature | Sequential | OpenMP | MPI | CUDA |
|---|---|---|---|---|
| Computing model | Sequential execution | Shared-memory parallelism | Distributed-memory parallelism | GPU parallelism |
| Main processing hardware | CPU | Multi-core CPU | CPUs across multiple processes/nodes | NVIDIA GPU |
| Execution unit | Single CPU execution flow | CPU threads | MPI processes | GPU threads |
| Memory model | Shared CPU memory | Shared CPU memory | Separate memory per process | GPU device memory |
| Number of parallel units | 1 | 8 threads | 4 processes | 256 threads per block |
| Communication | Not required | Shared memory | Message passing | Host-device memory transfer |
| Synchronization | Minimal | Thread synchronization | MPI synchronization | GPU synchronization |
| Main overhead | Computation | Thread management | Communication and synchronization | Memory transfer and kernel launch |
| Scalability | Limited | Depends on CPU cores | Can extend across processes/nodes | Large number of GPU threads |
| Primary purpose in experiment | Baseline | CPU parallelism | Distributed parallelism | GPU parallelism |

---

# 3. Workload Specifications

## 3.1 Matrix Configuration

The same workload is used for all four implementations to make the performance comparison consistent.

```text
Matrix A = 4000 × 4000
Matrix B = 4000 × 4000
Matrix C = 4000 × 4000
```

The input matrices are initialized with:

```text
A[i][j] = 1.0
B[i][j] = 1.0
```

The result matrix is initially:

```text
C[i][j] = 0.0
```

For every element of the result matrix:

```text
C[i][j] = A[i][0] × B[0][j]
        + A[i][1] × B[1][j]
        + ...
        + A[i][3999] × B[3999][j]
```

Since every input element is 1.0 and there are 4000 terms:

```text
C[i][j] = 4000.00
```

Therefore, the expected verification value is:

```text
C[0][0] = 4000.00
```

This known expected value is used to check the correctness of the implementations.

## 3.2 Computational Complexity

Standard matrix multiplication uses three nested loops:

```text
for i = 0 to N-1
    for j = 0 to N-1
        for k = 0 to N-1
            C[i][j] += A[i][k] × B[k][j]
```

Therefore, its computational complexity is:

```text
O(N³)
```

For:

```text
N = 4000
```

the number of iterations of the innermost computation is:

```text
4000³ = 64,000,000,000
```

This large computational workload makes matrix multiplication suitable for demonstrating the effects of parallel computing.

## 3.3 OpenMP Configuration

The OpenMP implementation uses:

```text
Number of CPU threads = 8
```

The workload is divided among the eight threads by parallelizing the outer matrix loop.

Conceptually:

```text
4000 rows
    |
    +---- Thread 1
    +---- Thread 2
    +---- Thread 3
    +---- Thread 4
    +---- Thread 5
    +---- Thread 6
    +---- Thread 7
    +---- Thread 8
```

Each thread processes a portion of the matrix rows.

## 3.4 MPI Configuration

The MPI implementation uses:

```text
Number of MPI processes = 4
```

The workload is divided equally:

```text
4000 rows / 4 processes = 1000 rows per process
```

Therefore:

```text
Rank 0 → 1000 rows
Rank 1 → 1000 rows
Rank 2 → 1000 rows
Rank 3 → 1000 rows
```

The MPI setup consists of:

```text
Master
Worker 1
Worker 2
Worker 3
```

The processes communicate using MPI message-passing operations.

## 3.5 CUDA Configuration

The CUDA implementation uses the following configuration:

```text
Matrix size       = 4000 × 4000
Block dimensions  = 16 × 16
Threads per block = 256
Grid dimensions   = 250 × 250
Total blocks      = 62,500
```

The grid dimensions are calculated as:

```text
4000 / 16 = 250
```

Therefore:

```text
Grid = 250 × 250
```

and:

```text
250 × 250 = 62,500 blocks
```

Each block contains:

```text
16 × 16 = 256 threads
```

Each CUDA thread is responsible for computing one output element of the result matrix, subject to the boundary checks implemented in the kernel.
