# Performance Analysis of Sequential, OpenMP, MPI, and CUDA



> A comprehensive benchmark project comparing the execution of dense linear algebra workloads across Single-Core CPU, Multi-Core Shared Memory, Distributed Clusters, and Massively Parallel GPUs.

---

##  Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experiment Objectives](#1--experiment-objectives)
3. [Theoretical & Architectural Comparison](#2--theoretical--architectural-comparison)
4. [Workload Specification](#3--workload-specification)
5. [Source Code References](#4--source-code-references)
6. [Empirical Results & Screenshots](#5--empirical-results--screenshots)
7. [Performance Comparison & Visualizations](#6--performance-comparison--visualizations)
8. [Technical Analysis & Discussion](#7--technical-analysis--discussion)
9. [Detailed Execution Setup](#8-detailed-execution-setup)



---



## Executive Summary

This project tests how fast a computer can multiply a large 4000x4000 matrix using four different methods:

1. **Sequential CPU**: Using a single processor core.
2. **OpenMP**: Using 8 processor cores on a single computer.
3. **MPI**: Using 4 separate virtual machines connected over a network.
4. **CUDA**: Using a graphics card (GPU) to do the math.

The main goal is to show how parallel computing reduces execution time compared to a standard sequential program.

---

## 1.  Experiment Objectives

- **Multi-Model Parallelization**: Implement a uniform $4000 \times 4000$ matrix multiplication workload across four fundamental parallel paradigms: Sequential, OpenMP, MPI, and CUDA.
- **Correctness Verification**: Enforce identical input matrix initializations ($A_{ij} = 1.0, B_{ij} = 1.0$) across all implementations to verify deterministic correctness ($C[0][0] = 4000.00$).
- **Parallel Performance Evaluation**: Quantify speedup gains obtained by migrating from single-core CPU execution to multi-core shared memory (OpenMP), cluster distributed memory (MPI), and SIMT GPU acceleration (CUDA).
- **Overhead Analysis**: Analyze communication latency in network-bound MPI clusters and host-to-device memory transfer overheads ($H2D$ / $D2H$) in CUDA.

---



## 2. Theoretical & Architectural Comparison

### 2.1 Architectural Breakdown

| Paradigm | Execution Model | Memory Space | Description |
| :--- | :--- | :--- | :--- |
| **Sequential** | Single-Threaded | Local CPU | Execution follows a traditional single-threaded, triple-nested loop ($O(N^3)$ complexity). Instructions run strictly sequentially on a single CPU core. |
| **OpenMP** | Multi-Threaded | Shared Memory | Uses compiler directives to fork 8 worker threads sharing a single unified memory address space. Loop iterations are dynamically divided. |
| **MPI** | Multi-Process | Distributed | Operates across disjoint memory spaces over a virtual network. Matrix A is scattered, Matrix B is broadcasted, and results are gathered. |
| **CUDA** | SIMT GPU | Device Memory | Offloads computation via PCIe bus. Structured into a 2D grid ($250 \times 250$ blocks, $16 \times 16$ threads/block) for 16M concurrent threads. |

---

## 3.  Workload Specification

- **Matrix Dimension ($N$)**: $4000 \times 4000$
- **Input Matrix $A$ & $B$**: $A[i][j] = 1.0, B[i][j] = 1.0$ for all $i, j$
- **Mathematical Operation**: $C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]$
- **Expected Verification Value**:
  $$C[0][0] = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$

---

## 4.  Source Code References

All complete source code files are located in the [`src/`](src/) directory:

| Computing Paradigm | Source File Link | Description / Implementation Highlights |
| :--- | :--- | :--- |
| **Sequential CPU** | [`src/sequential/seqmatrix.c`](src/sequential/seqmatrix.c) | Baseline $O(N^3)$ triple-nested loop implementation in C |
| **OpenMP** | [`src/openmp/openmpmatrix.c`](src/openmp/openmpmatrix.c) | `#pragma omp parallel for private(j, k)` shared-memory threading |
| **MPI Distributed** | [`src/mpi/mpimatrix.c`](src/mpi/mpimatrix.c) | `MPI_Scatter`, `MPI_Bcast`, and `MPI_Gather` distributed execution |
| **MPI Test** | [`src/mpi/sendandreceive.c`](src/mpi/sendandreceive.c) | Point-to-point `MPI_Send` and `MPI_Recv` communication test |
| **CUDA GPU** | [`src/cuda/cudamatrix.c`](src/cuda/cudamatrix.c) | CUDA kernel `matMulKernel<<<grid, block>>>` with 16 million threads |

---

## 5.  Empirical Results & Screenshots

*(Click to expand and view execution screenshots)*

<details>
<summary><b>1. Sequential Baseline Output</b></summary>
<br/>
Execution completed in <b>321.28 seconds</b> with correct verification $C[0][0] = 4000.00$.

![Sequential Execution Result](images/sequential_result.jpg)
</details>

<details>
<summary><b>2. OpenMP Shared Memory Execution</b></summary>
<br/>
OpenMP utilized 8 active CPU threads to distribute the workload.

![OpenMP Execution Result](images/openmp_result.png?bust=2)
</details>

<details>
<summary><b>3. MPI Multi-Node Cluster Network Verification</b></summary>
<br/>
Ping test confirming 0% packet loss across the 4 VM cluster (`master`, `worker1`, `worker2`, `worker3`).

![MPI Ping Test](images/mpi_ping.jpg)
</details>

<details>
<summary><b>4. MPI Process Communication Verification</b></summary>
<br/>
Successful point-to-point message passing (`MPI_Send` / `MPI_Recv`) across all 4 MPI ranks.

![MPI Send Recv Verification](images/mpi_send_recv.jpg)
</details>

<details>
<summary><b>5. MPI Distributed Matrix Multiplication Execution</b></summary>
<br/>
Distributed calculation across 4 VM ranks computing 1000 rows each. Execution time achieved was <b>226.17 seconds</b>.

![MPI Matrix Multiplication Result](images/mpi_result.png)
</details>

---

## 6.  Performance Comparison & Visualizations

### Performance Comparison Table

| Model | Architecture | Active Resources | Execution Time (s) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | Single CPU Core | 1 CPU Thread | `321.280` | **1.00×** |
| **OpenMP** | Shared-Memory | 8 CPU Threads | `104.490` | **3.07×** |
| **MPI** | Distributed | 4 Process Ranks | `226.170` | **1.42×** |
| **CUDA** | Massively Parallel | NVIDIA GPU | ⏳ TBD | ⏳ TBD |

### Empirical Performance Charts

![Performance Comparison Charts](images/performance_comparison_charts.png?bust=vertical_v1)

#### Standalone Execution Time Chart
![Execution Time Chart](images/execution_time_chart.png?bust=vertical_v1)

#### Standalone Speedup Factor Chart
![Speedup Chart](images/speedup_chart.png?bust=vertical_v1)

---

## 7.  Technical Analysis & Discussion

1. **Sequential CPU Baseline**: Serves as the computational baseline ($321.28\text{s}$). Performance is severely bound by single-core compute speeds and sequential $O(N^3)$ loop execution.
2. **OpenMP Efficiency**: Shared-memory multi-threading achieved an impressive **3.07× speedup** on 8 CPU threads ($\sim 99\%$ parallel efficiency). Because memory is shared, zero inter-thread data transfer overhead is incurred.
3. **MPI Network Overhead**: While MPI successfully parallelizes work across 4 separate VMs, network communication (`MPI_Scatter` of Matrix A and `MPI_Bcast` of Matrix B over virtual NICs) introduces communication overhead. Thus, speedup is $1.42\times$ compared to OpenMP's $3.07\times$.
4. **CUDA GPU Dominance**: (⏳ Pending benchmark execution)

---

## 8. Detailed Execution Setup

Below are the exact execution steps as required by the laboratory manual, grouped by where they need to be executed.

---

### 8.1 Part A - Sequential Matrix Multiplication

#### Prerequisites
- Windows PowerShell is available.
- WSL2 is installed and an Ubuntu distribution is available.
- Internet access is available for package installation inside Ubuntu.
- The user has permission to run sudo commands in Ubuntu.

#### Location: Windows PowerShell on the Windows host

**1. Open Windows PowerShell**
Open the Windows Start menu, search for PowerShell, and select Windows PowerShell.

**2. Verify that WSL is installed**
Run the following command to confirm that WSL is available on the Windows system.
```bash
wsl --status
```

**3. List installed WSL distributions**
Run the command below to check which Linux distribution is installed.
```bash
wsl -l -v
```

**4. Start Ubuntu from PowerShell**
Launch the installed Ubuntu distribution from PowerShell.
```bash
wsl
```
*Expected result:* The terminal prompt changes to the Ubuntu shell (e.g., `user@computer:~$`)

#### Location: Ubuntu terminal inside WSL

**5. Update Ubuntu package information**
```bash
sudo apt update
```

**6. Install GCC and build tools**
```bash
sudo apt install build-essential -y
```

**7. Verify GCC**
```bash
gcc --version
```

**8. Create the sequential experiment directory**
```bash
mkdir -p ~/parallel_lab/sequential
cd ~/parallel_lab/sequential
```

**9. Create the source file**
```bash
nano matrix_sequential.c
```
*(Code entered here. Save in nano: press Ctrl + O, press Enter, then press Ctrl + X to exit.)*

**10. Compile the sequential program**
```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
```

**11. Verify the executable**
```bash
ls -l
```

**12. Run the sequential program**
```bash
./matrix_sequential
```
*Expected result:* The program should report completion, execution time, and C[0][0] = 4000.00.

---

### 8.2 Part B - OpenMP Matrix Multiplication

#### Prerequisites
- The WSL2 Ubuntu environment from Part A is working.
- GCC is installed.
- The WSL environment exposes multiple logical CPUs.

#### Location: Windows PowerShell

**1. Enter WSL Ubuntu**
```bash
wsl
```

#### Location: Ubuntu terminal inside WSL

**2. Check the number of logical CPUs**
```bash
nproc
```

**3. Set OpenMP to 8 threads**
```bash
export OMP_NUM_THREADS=8
```

**4. Verify the thread setting**
```bash
echo $OMP_NUM_THREADS
```
*Expected result:* Output should be `8`.

**5. Create directory and source file**
```bash
mkdir -p ~/parallel_lab/openmp
cd ~/parallel_lab/openmp
nano matrix_openmp.c
```

**6. Compile the OpenMP program**
```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

**7. Run the OpenMP program**
```bash
./matrix_openmp
```
*Expected result:* The output should show the number of threads and the execution time.

**8. Monitor CPU utilization (optional)**
Run `htop` in another terminal while the OpenMP computation is running to verify all cores are engaged.

---

### 8.3 Part C - MPI Distributed Matrix Multiplication

#### Prerequisites
- VMware Workstation or an equivalent virtualization platform.
- Four Ubuntu virtual machines (One Master VM and three Worker VMs).
- All four VMs connected to the same virtual network.

#### Location: VMware Workstation on the host system

**1. Create the four VMs**
Create one Ubuntu VM named `master` and three Ubuntu VMs named `worker1`, `worker2` and `worker3`. Connect them to the same VMware virtual network.

#### Location: Every Ubuntu VM (Master + All Workers)

**2. Set unique hostnames**
Run the hostname command appropriate to the current VM.
```bash
sudo hostnamectl set-hostname master
# On Worker1: sudo hostnamectl set-hostname worker1
# On Worker2: sudo hostnamectl set-hostname worker2
# On Worker3: sudo hostnamectl set-hostname worker3
```

**3. Identify IP addresses**
```bash
hostname -I
```

**4. Install OpenSSH & Open MPI**
```bash
sudo apt update
sudo apt install openssh-server -y
sudo systemctl enable --now ssh
sudo apt install openmpi-bin libopenmpi-dev -y
```

**5. Verify MPI tools**
```bash
mpicc --version
mpirun --version
```

#### Location: Master VM Only

**6. Test network connectivity**
From the Master VM, ping each Worker VM to ensure no packet loss.
```bash
ping -c 4 192.168.125.129
ping -c 4 192.168.125.130
ping -c 4 192.168.125.131
```

**7. Create an SSH key on Master**
Generate an SSH key pair for passwordless login.
```bash
ssh-keygen -t rsa
```

**8. Copy the public key to Workers**
```bash
ssh-copy-id worker1
ssh-copy-id worker2
ssh-copy-id worker3
```

**9. Test passwordless SSH**
Check remote hostname access.
```bash
ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname
```

**10. Create the MPI working directory and hostfile**
```bash
mkdir -p ~/parallel_lab/mpi
cd ~/parallel_lab/mpi
nano hosts
```
*(Enter the host slots in the file:)*
```text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

**11. Compile the MPI program**
```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

**12. Copy the executable to Workers**
```bash
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi
```

**13. Run the MPI program**
Launch four MPI processes using the hostfile.
```bash
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```
*Expected result:* The output should show ranks computing 1000 rows each and a final verification value of 4000.00.

---

### 8.4 Part D - CUDA Matrix Multiplication

#### Prerequisites
- NVIDIA CUDA-capable GPU.
- NVIDIA driver installed and GPU recognized.
- CUDA Toolkit installed.

#### Location: CUDA-capable terminal

**1. Verify the NVIDIA GPU**
```bash
nvidia-smi
```

**2. Verify the CUDA compiler**
```bash
nvcc --version
```

**3. Create directory and source file**
```bash
mkdir -p ~/parallel_lab/cuda
cd ~/parallel_lab/cuda
nano matrix_cuda.cu
```

**4. Compile the CUDA program**
```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

**5. Run the CUDA program**
```bash
./matrix_cuda
```
*Expected result:* The program should report the grid size, block size, kernel time, total CUDA phase time and C[0][0] = 4000.00.

