---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
![[Pasted image 20260330160941.png]]

```c
unsigned int t = threadIdx.x;
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
	__syncthreads();
	if (t % (2*stride) == 0)
		partialSum[t] += partialSum[t+stride];
}
```

### Issues
Some threads perform addition while other do nothing (the odd index ones), so no more than half of threads will be executing at any time

---
## Better way of doing a reduce
![[Pasted image 20260330161218.png]]

>[!tldr]
>Intuition: for the first steps, all the threads in a warp will do the same thing (either sum, or do nothing)
>
>E.g., for a 512 threads block:
>- 1° step: the fist 512 threads read data from the other 512 threads (512 active threads)
>- 2° step: threads `[0, 255]` read data from `[256, 511]`
>- …
>- after the 5° step, the stride will $<32$, so we will have divergence (but only for the last few steps)

```c
_shared_ float partialSum[SIZE];
partialsum[threadIdx.x] = X[blockIdx.x*blockDim.x+threadIdx.x];

unsigned int t = threadIdx.x;
for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride>>1) {
	_syncthreads ();
	if (t < stride)
		partialSum[t] += partialSum[t+stride];
}
```

>[!question] What if $N>block\_size$?
>Shared memory only shared between threads in the same block. We need to reduce a larger vector than the number of threads per block
>
>Suppose you need to reduce an array of `v[N]` elements, with `N = Nblock*Nthread`. We need to:
>1. Use a kernel `reduce<<<Nblock,Nthread>>>(v,partial)` to get an array of `Nblock` partial values (`partial` is an array of `Nblock` elements allocated in global memory)
>2. Use a kernel `reduce<<<1,Nblock>>>(partial,result)` to get the final result

---
## Moving data between GPUs
There are 2 different solutions, based on whether MPI is GPU-aware or not:
- *MPI is not GPU-aware*: data must be transferred from device to host memory before making the desired MPI call (and viceversa for the receiving side)
- *MPI is GPU-aware*: MPI can access device buffers directly, hence pointers to device memory can be used in MPI calls
(on the cluster MPI is not GPU-aware)