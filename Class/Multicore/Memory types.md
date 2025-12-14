---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
There are multiple types of memories, some on-chip and some off-chip

![[Pasted image 20251214152036.png]]

Let’s analyze them:
- registers → holding local variables
- shared memory → fast on-chip memory to hold frequently used data; can be also used to exchange data between the cores of the same SM
- global memory → main part of the off-chip memory that has high capacity by is relatively slow; it is the only part accessible by the host through CUDA functions
- texture and surface memory → content managed by special hardware that permits fast implementation of some filtering/interpolation operator
- constant memory → can only store constants; it is cached, and allows broadcasting of a single value to all threads in a warp (less appealing on newer GPUs that have a cache anyway)

---
## Shared memory
Shared memory is an on-chip memory, different from registers. In fact registers data is private to threads, while shared memory is shared among threads (can be seen as a user-manages L1 cache, a scratch pad)

It can be also used as:
- a place to hold frequently used data that would otherwise require a global memory access
- a way for cores on the same SM to share data

The `__shared__` specifier can be used to indicate that some data must go in the shared on-chip memory rather than on the global memory

>[!info] Shared memory vs. L1 cache
>- both are on-chip; the former is managed by the programmer the latter automatically
>- in some cases, managing it manually (i.e., using the shared memory), might provide better performance (e.g., you do not have any guarantee that the data you need will be in the L1 cache, but with the explicitly managed shared memory, you can control that)

>[!example] 1D stencil
>A 1D stencil is a local computation where the new value of an element in an array (or grid) is calculated based on its _current_ value and the values of its immediate neighbors.
>
>Consider applying a 1D stencil to a 1D array of elements (each output element is the sum of input elements within a radius)
>
>![[Pasted image 20251210145255.png|350]]
>If radius is 3, then each output element is the sum of 7 input elements
>
>Each thread processes one output element (`blockDim.x` elements per block), so input elements are read several times. In the current example, with radius 3 each input element is read seven times
>
>To avoid multiple readings we cache data in shared memory:
>- read $\verb|blockDim.x|+2\cdot \text{radius}$ input elements from global memory to shared memory
>- compute `blockDim.x` output elements
>- write `blockDim.x` output elements to global memory
>
>By applying this rule, we need to add a halo (padding) of `radius` elements at the beginning and at the end of our “array” to make sure that every element can compute the output value
>
>![[Pasted image 20251214160536.png]]
>
>```c
>__global__ void stencil_1d(int *in, int *out) {
>	// adding padding at the beginning and ath the end of the array
>	// (2*RADIUS)
>	// BLOCK_SIZE -> number of elements for each thread
>	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
>	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
>	// in position threadId.x I'm not considering halo
>	// need to add radius
>	int lindex = threadIdx.x + RADIUS;
>	
>	// read input elements into shared memory
>	temp[lindex] = in[gindex];
>	if (threadIdx.x < RADIUS) {
>		temp[lindex-RADIUS] = in[gindex-RADIUS];
>		temp[lindex+BLOCK_SIZE] = in[gindex+BLOCK_SIZE]
>	}
>	
>	__syncthreads();
>	
>	// apply the stencil
>	int result = 0;
>	for (int offset=-RADIUS; offset<=RADIUS; offset++)
>		result += temp[lindex+offset];
>	
>	// store the result
>	out[gindex] = result;
>}
>```

