---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
It enables a general-purpose programming model on NVIDIA GPUs: before CUDA, GPU was programmed by transofrming an algorithm in a sequence of image manipulation primitives

Enables explicit GPU memory management and it is viewed as a compute device that:
- is a co-processor to the CPU (or host)
- has its own DRAM (global memory in CUDA parlance)
- runs many threads in parallel: thread creation/switching cost is few clock cycles

>[!warning]
>CUDA is a platform/programming model, not a programming language

---
## Program structure
A common CUDA program follows these steps:
1. allocate GPU memory
2. transfer data from host to GPU memory
3. run CUDA kernel (the function that is being executed on the GPU)
4. copy result from GPU memory to host memory (to be able to use it, I have to move it to the host memory once again)

---
## Execution model
![[Pasted image 20251125174829.png]] ^b1cc83

In the vast majority of scenarios, the host is responsible for I/O operations, passing the input and subsequently collecting the output data from the memory space of the GPU

![[Pasted image 20251125174939.png]] ^example-grid

CUDA organized threads in a 6D structure (lower dimensions are also possible).
Each thread has a position in a 1D, 2D or 3D *block*. Each block has a position in a 1D, 2D, or 3D *grid*. Each thread is aware of its position in the overall structure, via a set of intrinsic variables/structures.

With this information a thread can map its position to the subset od data that is assigned to.

>[!example]
>In the [[#^example-grid|picture]] above each block is 2D, while the grids are 3D.

>[!info]
>We choose the structure based on what we are working on (e.g. for images 2D grid and 1D block, for fluids 3D grid, etc.)

### Compute capability
The sizes of blocks and grids are determined by the capability, which determined what each generation of GPUs is capable of and the compute capability of a device is represented by a version number, also sometimes called its “SM version”. This version number identifies the feature supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU

![[Pasted image 20251125175932.png]]

![[Pasted image 20251125175959.png]]

---
## How to write a program?
You must specify a function that is going to be executed by all the threads (SIMD/SPMD/SIMT). This function is called **kernel**

You must specify how threads are arranged in the grid/blocks. The following example is based on the above [[#^example-grid|picture]]

```c
// if you don't specify the other dimension, are set by default to 1
dim3 block(3,2);
dim3 grid(4,3,2)

// acrivates the CUDA kernel
foo<<<grid. block>>>();
```

Let’s analyze the parameters:
- `dim3` is a vector of `int`
- every non-specified component is set to $1$
- every component accessible to $x$, $y$, $z$ fields (will se it later)

### How to specify grid/block size

```c
dim3 b(3,3,3);
dim3 g(20,100);
foo<<<g. b>>>(); // run a 20x100 grid of 3x3x3 blocks
foo<<<10, b>>>(); // run a 10 block grid, each made by 3x3x3 threads
foo<<<g, 256>>>(); // run a 20x100 grid, made of 256 threads
foo<<<g, 2048>>>(); // an invalid example: max block size is 1024 threads even for compiute capability 5.x
foo<<<5, g>>>(); // another invalid example that specifies a block size of 20x100=2000 threads
```

---
## Hello world in CUDA

```c
// file hello.cu
#include <stdio.h>
#include <cuda.h>

// can be called from the host or the device and must run on the device (GPU)
// a kernel is always declared as void and computed result must be copied explicitly from GPU to host memory to be able to access them
__global__ void hello() {
	// supported by CC 2.0 (compute capability)
	printf("Hello world!\n");
}

int main() {
	hello<<<1,10>>>();
	// blocks until the CUDA kernel terminates
	cudaDeviceSyncronize();
	return 1;
}
```

^hello-world

To compile and run:
```bash
# arch specifica la capability della GPU
$ nvcc --arch=sm_20 hello.cu -o hello
$ ./hello
```

>[!warning]
>`printf` is detrimental for performance (GPU should not do I/O). Only use it for debugging purposes

### Function decoration
- `__global__` → can be called from the host or the GPU and executed on the device/GPU. In CC 3.5 and above, the device can also call `__global__` functions
- `__device__` → a function that runs on the GPU and can only be called from within a kernel (i.e. from the GPU)
- `__host__` → a function that can only run on the host. The `__host__` qualifier is typically omitted, unless used in combination with `__device__` to indicate that the function can run on both the host and the device. Such scenario implies the generation of two compiled codes for the function

---
## Determine the thread position in the grid/block

>[!example]
>![[Pasted image 20251125182427.png]]
>
>$$x=\verb|blockIdx.x|\times \verb|blockDim.x|+\verb|threadIdx.x|=1\times 4+3=7$$
>$$y=\verb|blockIdx.y|\times \verb|blockDim.y|+\verb|threadIdx.y|=2\times 4+1=9$$
>$$\verb|threadId|=9\times 16+7=151$$
>
>Where `threadId` is the absolute position, not considering block and grid division

>[!example]
>![[Pasted image 20251126113317.png]]
>
>$$\verb|index|=\verb|blockIdx.x|\cdot \verb|blockDim.x| + \verb|threadIdx.x|=2\cdot 256 + 3=515$$

>[!info] How to get a unique thread id in a generic structure?
>Different threads might have the same `threadIdx` but be on different blocks, so I need to combine `threadIdx` and `blockIdx` to get a unique identifier
>
>```c
>int myID = ( blockIdx.z * gridDim.x * gridDim.y +
>            blockIdx.y * gridDim.x +
>            blockIdx.x ) * blockDim.x * blockDim.y * blockDim.z +
>            threadIdx.z * blockDim.x * blockDim.y +
>            threadIdx.y * blockDim.x +
>            threadIdx.x;
>```
>
>Often threads are arranged in fewer than 6 dimensions (i.e. some of those dimensions will be equal to 1 and the corresponding coordinates to 0)
>
>>[!example]
>>To get the ID for the [[#^hello-world|hello world case]] (threads were arranged in 1 block of 10 threads):
>>```c
>>int i = blockIdx.x * blockDim.x + threadIdx.x;
>>```

---
## Thread scheduling
![[Pasted image 20251125172610.png]]

Each thread runs on a streaming processor (CUDA core) and sets of cores on the same SM share the control unit (i.e. they must synchronously execute the same instruction).

Different SMs can run different kernels and each block runs on an SM (i.e. I can’t have a block spanning over multiple SMs, but I can have more blocks running on the same SM). Once a block is fully executed, the SM will run the next one.

>[!warning]
>Not all the threads in a block run concurrently

---
## Warps
Threads are executed in groups called **warps** (in current GPUs, the size of warp is 32, but might change in future). Threads in a block are split into warps according to their intra-block ID (i.e. the first 32 threads in a block belong to the same warp, the next 32 threads to a different warp, etc.)

All threads in a warp are executed according to Single Instruction, Multiple Data (SIMD) model (i.e. at any instant in time, one instruction is fetched and executed for all threads in the warp). Consequently, all threads in a warp will always have the same execution timing.

Several warp scheduler (e.g. 4) can be present on each SM (i.e. multiple warps can run at the same time, each possibly following a different execution path)

![[Pasted image 20251126120122.png|center|350]]

### Warp divergence

>[!question] What happens if the result of conditional operation in a warp leads to different paths?

The thread that do not follow the path currently being executed are stalled

![[Pasted image 20251126120328.png]]

### Context switching
Usually a SM has more resident blocks/warps than what it is able to concurrently run and each SM can switch seamlessly between warps. In fact each thread has its own private **execution context that is maintained on-chip** (i.e. context switch comes for free)

When an instruction to be executed by a warp need to wait for the result of a previously initiated long-latency operation, the warp is not selected for execution (e.g. memory read, long latency floating-point operations). This mechanism of filling the latency time of operations with work from other threads often called **latency tolerance** or *latency hiding*

Given a sufficient number of warps, the hardware will likely find a warp to execute at any point in time, thus making full use of the execution hardware in spite of these long latency operations

>[!info]
>With warp scheduling, the long waiting time of warp instruction is “hidden” by executing instructions from other warps

This ability to tolerate long-latency operations is the main reason GPUs do not dedicate nearly as much chip area to cache memories and branch prediction mechanisms as do CPUs

>[!example] 
>>[!question] A CUDA device allows up to 8 blocks and 1024 threads per SM, and 512 threads per block. Shall we use 8x8, 16x16 or 32x32 thread block?
>
>>[!done] 8x8 blocks
>>We would have 64 threads per block, so to fill the 1024 threads we can have for each SM, we  would need $1024/64=16$ blocks
>>
>>However, we can have at most 8 blocks per SM, thus we would end with only $64\times 8=512$ threads per SM so we are not fully utilizing the resources.
>>Most likely, at some time, the scheduler might not find threads to schedule when some thread is waiting for long-latency operations
>
>>[!done] 16x16 blocks
>>We would have 256 threads per block, so to fill the 1024 threads we can have for each SM, we would need $1024/256=4$ blocks.
>>
>>This would allow to have 1024 threads on the SM, so there will be a lot of opportunities for latency hiding
>
>>[!done] 32x32 blocks
>>We would have 1024 threads per block, which is higher than the 512 threads per block we can have

>[!example] 
>>[!question] A CUDA device allows up to 8 blocks and 1536 threads per SM, and 1024 threads per block. Shall we use 8x8, 16x16 or 32x32 thread block?
>
>>[!done] 8x8 blocks
>>We would have 64 threads per block, so to fill the 1536 threads we can have for each SM, we would need $1536/64 = 24$ blocks and $24>8$ so it can’t be possible
>
>>[!done] 16x16 blocks
>>We would have 256 threads per block, so to fill the 1536 threads we can have for each SM, we would need $1536/256 = 6$ blocks
>>
>>We achieve full capacity (unless other resources constraints come into play)
>
>>[!done] 32x32 blocks
>>We would have 1024 threads per block. Only one block can fit (two would bring the number of threads to $2048$, which is higher than $1536$)
>>
>>Thus, we would use only $2/3$ of the thread capacity of the SM ($1024$ out of $1536$)

>[!example] 
>A grid of 4x5x3 blocks, each made of 100 threads and the GPU has 16 SMs. Thus, we have 4x5x3=60 blocks, that need to be distributed over 60 16SMs
>
>Let’s assume that they are distributed round-robin and 12 SMs will receive 4 blocks, and 6 SMs will receive 3 blocks. It would be inefficient, in fact while the first 12 SMs process the last block, the other 6 SMs are idle
>
>A block contains 100 threads, which are divided into $100/32 = 4$ warps. In this case the first three warps have $32$ threads, and the last one have $4$ threads ($32+32+32+4$)
>Let’s assume we can only schedule a warp at a time (e.g., because we have 32 CUDA cores per SM), so the last warp would only use 4 out of the 32 available cores ($87.5\%$ of the cores on each SM will be unused)