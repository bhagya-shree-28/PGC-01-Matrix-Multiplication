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
   - [6.1 Execution Time](#61-execution-time)
   - [6.2 Correctness Verification](#62-correctness-verification)
   - [6.3 Speedup](#63-speedup)
   - [6.4 Result Summary](#64-result-summary)
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

---

# 4. Repository Structure

```text
Parallel-Matrix-Multiplication/
├── README.md
├── .gitignore
│
├── sequential/
│   └── matrix_sequential.c
│
├── openmp/
│   └── matrix_openmp.c
│
├── mpi/
│   ├── matrix_mpi.c
│   └── hosts
│
├── cuda/
│   └── matrix_cuda.cu
│
└── results/
    ├── execution_time.png
    ├── speedup.png
    └── screenshots/
```

---

# 5. Execution Steps

The following steps describe how each implementation was compiled and executed. The same matrix multiplication workload was used for all four implementations so that their execution times could be compared fairly.

## 5.1 Sequential Implementation

The sequential version provides the baseline execution time for comparison.

### Step 1: Enter the WSL Ubuntu environment

```bash
wsl
```

Used to enter the Linux environment where the C program is compiled and executed.

### Step 2: Verify the GCC compiler

```bash
gcc --version
```

Used to confirm that GCC is installed and available for compiling the C program.

### Step 3: Create and enter the sequential directory

```bash
mkdir -p ~/parallel_lab/sequential
cd ~/parallel_lab/sequential
```

Creates a separate working directory for the sequential implementation and moves into it.

### Step 4: Compile the program

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
```

Compiles the sequential C program.

- `-O2` enables compiler optimizations to improve execution performance.
- `-o` specifies the name of the generated executable.

### Step 5: Execute the program

```bash
./matrix_sequential
```

Runs the sequential matrix multiplication program. The measured execution time is used as the baseline for calculating speedup.

---

## 5.2 OpenMP Implementation

The OpenMP version uses multiple CPU threads to perform matrix multiplication in parallel.

### Step 1: Check available CPU processors

```bash
nproc
```

Displays the number of available CPU processing units. This helps determine an appropriate number of OpenMP threads.

### Step 2: Set the number of OpenMP threads

```bash
export OMP_NUM_THREADS=8
```

Sets OpenMP to use 8 CPU threads for the parallel computation.

### Step 3: Verify the thread configuration

```bash
echo $OMP_NUM_THREADS
```

Confirms that the OpenMP environment variable is set to 8.

### Step 4: Compile the OpenMP program

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

Compiles the C program with optimization.

- `-fopenmp` enables OpenMP support and links the required OpenMP library.

### Step 5: Execute the program

```bash
./matrix_openmp
```

Runs the matrix multiplication using multiple CPU threads. The execution time is recorded for comparison with the sequential version.

---

## 5.3 MPI Implementation

The MPI implementation distributes the matrix multiplication workload among multiple MPI processes.

In this experiment, 4 MPI processes were used:

```text
Rank 0 → Master   → 1000 rows
Rank 1 → Worker 1 → 1000 rows
Rank 2 → Worker 2 → 1000 rows
Rank 3 → Worker 3 → 1000 rows
```

### Step 1: Verify communication between machines

```bash
ping -c 4 worker1
ping -c 4 worker2
ping -c 4 worker3
```

Verifies that the Master can communicate with all worker machines before starting the MPI computation.

### Step 2: Install OpenMPI

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev -y
```

Installs the OpenMPI runtime and development libraries required to compile and execute MPI programs.

### Step 3: Generate an SSH key

```bash
ssh-keygen -t rsa
```

Generates an SSH key that can be used for passwordless communication between the Master and worker machines.

### Step 4: Copy the SSH key to the workers

```bash
ssh-copy-id worker1
ssh-copy-id worker2
ssh-copy-id worker3
```

Allows the Master machine to connect to the worker machines without repeatedly entering a password.

### Step 5: Configure the MPI host file

Create a file named `hosts`:

```text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

Specifies the machines that participate in the MPI computation and the number of MPI slots available on each machine.

### Step 6: Compile the MPI program

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

Compiles the MPI C program using the MPI compiler wrapper. `mpicc` automatically links the required MPI libraries.

### Step 7: Copy the executable to worker machines

```bash
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi
```

Copies the executable to each worker so that every MPI process can execute the same program.

### Step 8: Run the MPI program

```bash
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

Starts 4 MPI processes across the configured machines. The matrix rows are divided among the MPI processes. MPI communication is used to distribute data and collect results.

The MPI implementation uses operations such as:

- `MPI_Bcast`
- `MPI_Scatter`
- Local Matrix Multiplication
- `MPI_Gather`

Their roles are:

- `MPI_Bcast` distributes common matrix data.
- `MPI_Scatter` distributes portions of the workload.
- Each process performs multiplication on its assigned rows.
- `MPI_Gather` collects the partial results.

---

## 5.4 CUDA Implementation

The CUDA implementation performs matrix multiplication on an NVIDIA GPU. The CUDA program was compiled and executed in a MINGW64 (Git Bash) terminal on Windows.

### Step 1: Verify the NVIDIA GPU

```bash
nvidia-smi
```

Displays the installed NVIDIA GPU and its current status. This confirms that the GPU is available for CUDA execution.

### Step 2: Verify the CUDA compiler

```bash
nvcc --version
```

Confirms that the NVIDIA CUDA compiler is installed and available.

### Step 3: Enter the CUDA directory

```bash
cd ~/parallel_lab/cuda
```

Moves into the working directory that contains `matrix_cuda.cu`.

### Step 4: Compile the CUDA program

```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

Compiles the CUDA source file using `nvcc`.

- `-O2` enables compiler optimization.
- `-o` specifies the name of the generated executable.

### Step 5: Execute the CUDA program

```bash
./matrix_cuda
```

Executes the CUDA matrix multiplication program. The CPU transfers input data to GPU memory, launches the CUDA kernel, and copies the resulting matrix back to CPU memory.

The program output was:

```text
CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Grid Size = 250 x 250 blocks
Block Size = 16 x 16 threads
Kernel Execution Time = 0.154827 seconds
Total CUDA Phase Time = 0.183416 seconds
Verification C[0][0] = 4000.00
```

The CUDA configuration used in this experiment is:

```text
Matrix size       : 4000 × 4000
Block size        : 16 × 16
Threads per block : 256
Grid size         : 250 × 250
Total blocks      : 62,500
```

Each CUDA thread is responsible for computing one element of the output matrix:

```text
C[row][column]
```

The execution consists of:

```text
Host Memory
     |
     | Host → Device Transfer
     v
GPU Device Memory
     |
     | CUDA Kernel
     v
Parallel Matrix Multiplication
     |
     | Device → Host Transfer
     v
Host Memory
```

The CUDA kernel time and total CUDA phase time are recorded separately. The difference between them covers the other measured phases, such as memory transfer.

---

# 6. Experimental Results

The performance of all four implementations was measured using the same `4000 × 4000` matrix multiplication workload.

<img width="960" height="600" alt="results" src="https://github.com/user-attachments/assets/0cb4c0eb-aff1-4df8-ade8-b5f3cef8d4af" />

<img width="1600" height="900" alt="result" src="https://github.com/user-attachments/assets/99ea8412-c830-452c-94c9-3705057d62ac" />

<img width="959" height="980" alt="10-result" src="https://github.com/user-attachments/assets/860c0fd3-91d7-4d90-9fe3-ae40af1b0949" />

<img width="527" height="294" alt="results" src="https://github.com/user-attachments/assets/fa6966f7-c049-4c35-abac-7fc01ad7a5eb" />


## 6.1 Execution Time

| Implementation | Configuration | Execution Time (seconds) |
|---|---|---:|
| Sequential | Single CPU execution | 606.987331 |
| OpenMP | 8 CPU threads | 40.574496 |
| MPI | 4 MPI processes | 223.691390 |
| CUDA | GPU execution | 0.183416 |

For CUDA, the measured times were:

```text
CUDA Kernel Time: 0.154827 seconds
CUDA Total Phase: 0.183416 seconds
```

The total CUDA phase time (used for comparison) includes the kernel execution and the other measured GPU phases, such as data transfer.

## 6.2 Correctness Verification

The programs initialized matrices `A` and `B` with `1.0`. Therefore, every element of the resulting matrix should theoretically be:

```text
4000.00
```

because each output element performs 4000 multiplications of `1.0 × 1.0`.

| Implementation | Expected Result | Reported Result | Status |
|---|---:|---:|---|
| Sequential | 4000.00 | 4000.00 | Verified |
| OpenMP | 4000.00 | 4000.00 | Verified |
| MPI | 4000.00 | 4000.00 | Verified |
| CUDA | 4000.00 | 4000.00 | Verified |

All four implementations produced the expected verification value for `C[0][0]`.

## 6.3 Speedup

Speedup is calculated relative to the sequential implementation:

```text
Speedup = Sequential Execution Time / Parallel Execution Time
```

Using the measured execution times:

| Implementation | Execution Time (s) | Speedup |
|---|---:|---:|
| Sequential | 606.987331 | 1.00× |
| OpenMP | 40.574496 | 14.96× |
| MPI | 223.691390 | 2.71× |
| CUDA (total phase) | 0.183416 | 3309.35× |

For reference, the speedup based on the CUDA kernel time alone (0.154827 s) is 3920.42×.

## 6.4 Result Summary

```text
Sequential
    Execution Time : 606.987331 s
    Speedup        : 1.00×

OpenMP
    Execution Time : 40.574496 s
    Speedup        : 14.96×

MPI
    Execution Time : 223.691390 s
    Speedup        : 2.71×

CUDA
    Kernel Time    : 0.154827 s
    Total Phase    : 0.183416 s
    Speedup        : 3309.35× (total phase)
    Verification   : Verified (4000.00)
```

These measurements are used in the following sections to compare the performance characteristics of sequential, shared-memory, distributed-memory, and GPU-based parallel execution.


---

# 7. Performance Comparison

The measured results from all four implementations are compared using execution time and speedup.

## 7.1 Speedup

Speedup is calculated relative to the sequential baseline:

```text
Speedup = Sequential Execution Time / Parallel Execution Time
```

Parallel efficiency for the CPU-based implementations is calculated as:

```text
Efficiency = Speedup / Number of Parallel Units
```

| Implementation | Parallel Units | Execution Time (s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| Sequential | 1 | 606.987331 | 1.00× | 100% |
| OpenMP | 8 threads | 40.574496 | 14.96× | 187% |
| MPI | 4 processes | 223.691390 | 2.71× | 68% |
| CUDA | GPU threads | 0.183416 | 3309.35× | Not applicable |

> **Note:** The CUDA speedup uses the total CUDA phase time. Efficiency is not given for CUDA because the number of GPU threads is not comparable to CPU threads or processes. The CUDA run was executed in a native Windows environment (MINGW64), while the CPU runs were executed in WSL.

## 7.2 Execution Time Comparison

<img width="600" alt="execution_time" src="results/execution_time.png" />

This graph compares the execution time of all four implementations. A logarithmic scale is used because the CUDA time is much smaller than the sequential time.

## 7.3 Speedup Comparison

<img width="600" alt="speedup" src="results/speedup.png" />

This graph compares the speedup of OpenMP, MPI, and CUDA relative to the sequential baseline.

---

# 8. Technical Analysis

The analysis below is based on the measurements recorded in this experiment.

### 1. Sequential

The sequential implementation took **606.987331 seconds** for `4000³ = 64,000,000,000` multiply-add operations. It runs on a single CPU execution flow, so it is used as the baseline.

### 2. OpenMP

OpenMP reduced the execution time to **40.574496 seconds**, a speedup of **14.96×** with 8 threads.

- All threads share the same memory, so no data copying between threads is needed.
- Each thread processes different rows of `C`, so the iterations are independent.
- A speedup above 8× (superlinear) is higher than the ideal value for 8 threads. This suggests the sequential baseline is slower than expected, for example due to cache behavior from the `B[k][j]` access pattern, or from differences in loop order or compiler settings between the two programs. The baseline should be re-run to confirm this.

### 3. MPI

MPI reduced the execution time to **223.691390 seconds**, a speedup of **2.71×** with 4 processes, which is an efficiency of about 68%.

- Each process has its own memory, so data must be sent between processes.
- `MPI_Bcast` sends matrix `B` (about 128 MB in `double` precision) to every process, and `MPI_Scatter` and `MPI_Gather` move the rows of `A` and `C`.
- Because the processes run on separate machines connected by a network, communication time adds to the computation time.
- Synchronization between processes also adds overhead.

### 4. CUDA

CUDA recorded a kernel time of **0.154827 seconds** and a total phase time of **0.183416 seconds**, a speedup of **3309.35×** over the sequential baseline.

- The GPU runs a very large number of threads at the same time (62,500 blocks of 256 threads), with each thread computing one output element.
- The matrix has `4000 × 4000 = 16,000,000` output elements, so the GPU has enough independent work to keep its cores busy.
- The difference between the total phase and the kernel time (about 0.029 seconds) represents the other measured phases, such as memory transfer between host and device. This is about 16% of the total phase time.
- The verification value `4000.00` matches the expected result.

### 5. Overall Observation

| Implementation | Main Factor Affecting Performance |
|---|---|
| Sequential | Only one execution flow performs all computation |
| OpenMP | Shared memory gives low overhead, but the number of CPU cores is limited |
| MPI | Communication and synchronization between processes |
| CUDA | Large number of GPU threads, with host-device memory transfer as the main overhead |

The ranking by execution time is CUDA, then OpenMP, then MPI, then Sequential. These observations are specific to this hardware, workload, and configuration and should not be generalized to all systems.

---

# 9. Conclusion

This experiment implemented the same `4000 × 4000` matrix multiplication using Sequential, OpenMP, MPI, and CUDA approaches and compared their measured execution times.

The results showed:

- Sequential execution took 606.987331 seconds and was used as the baseline.
- OpenMP with 8 threads took 40.574496 seconds (14.96× speedup).
- MPI with 4 processes took 223.691390 seconds (2.71× speedup).
- CUDA took 0.183416 seconds for the total phase (3309.35× speedup), with a kernel time of 0.154827 seconds.
- All four implementations produced the expected verification value of `4000.00`.

CUDA gave the best performance because the GPU executes a very large number of threads in parallel. OpenMP was the best CPU-based approach because threads share memory and have low communication overhead. MPI gave a smaller speedup, which is consistent with the cost of communication between processes.

The experiment also provides practical experience with:

- Shared-memory, distributed-memory, and GPU-based parallel programming
- Compiling and running OpenMP, MPI, and CUDA programs
- Measuring execution time and calculating speedup
- Verifying results and analyzing performance overheads
