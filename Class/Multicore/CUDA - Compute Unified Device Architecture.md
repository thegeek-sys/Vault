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

You must specify how threads are arranged in the grid/blocks. The following example is based on the above [[#^17831c|picture]]

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

// can be called from the host or the device. must run on the device (GPU)
// it's a decorator
// a kernel is always declared as void. computed result must be copied explicitly from GPU to host memory
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