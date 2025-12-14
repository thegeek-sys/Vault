---
Class: "[[Multicore]]"
Related:
---
---
## Index
- [[#Shared memory|Shared memory]]
- [[#Constant memory|Constant memory]]
- [[#Performance estimation|Performance estimation]]
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
>	// make sure that every thread has loaded the data, so
>	// that the for does not read ghost data
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
>
>I need to execute `__syncthreads()` as I have no guarantee on the threads order. In particular I may have problems if `lindex+RADIUS` contains the halo (if a thread wants 32nd element but the thread that had to load that element haven’t already been executed)

```c
void __syncthreads();
```

Synchronizes all threads within a block. It is used to prevent RAW/WAR/WAW hazards

---
## Constant memory
Constant memory is not the ROM, it is just a memory that can hold constant values (i.e., the host can write it, but for the GPU is read-only)

It has two main advantages:
- it is cached
- supports broadcasting of a single value to all threads in a warp

>[!example]
>Suppose to have 10 warps on an SM, and all request the same variable:
>- if data is on global memory
>	- all warp will request the same segment from global memory
>	- the first time segment is copied into L2 cache
>	- if other data pass through L2, there are good chances it will be lost
>	- there are good chances that data should be requested multiple times
>- if data is in constant memory
>	- during first warp request, data is copied in constant-cache
>	- since there is less traffic in constant-cache, there are good chances all other warp will find the data already in cache


```c
__constant__ type variable_name; // static
cudaMemcpyToSymbol(variable_name, &host_src, sizeof(type), cudaMemcpyHostToDevice);
// warning: cannot be dynamically allocated
```

By saving variables in constant memory:
- data will reside in the constant memory address space
- has static storage duration (persists until the application ends)
- readable from all threads of a kernel

>[!example] Image to greyscale
>An image can be seen as a 2D matrix of pixels, each pixel has 3 values for the 3 RGB channels. 
>$$\text{Number of columns} = 3 \times  \text{Number of pixels in a row}$$
>
>To convert the pixel to grey scale:
>$$L=r\cdot 0.21 + g \cdot 0.72 + b\cdot 0.07$$
>
>The pixels can be calculated independently from each other
>![[Pasted image 20251214171255.png]]
>
>>[!warning]
>>![[Pasted image 20251214171430.png|Convering a $76 \times 62$ picture with $16 \times 16$ blocks]]
>>Some threads in some blocks (those in the 2, 3, 4 areas) do not have any pixel to process
>>Each threads should check if it has a pixel to process or not
>
>Before writing the code let’s see what we will have.
>![[Pasted image 20251214171959.png]]
>
>After assigning the `Col` and `Row` we will need to linearize the 2D matrix and to get the corresponding location for each element
>![[Pasted image 20251214172138.png]]
>
>```c
>// we have 3 channels corresponding to RGB
>// the input image is encoded as unsigned characters [0, 255]
>__global__ void colorToGreyScale(unsigned char* Pout, unsigned 
>								char* Pin int width, int height) {
>	int Col = threadIdx.x + blockIdx.x * blockDim.x;
>	int Row = threadIdx.y + blockIdx.y * blockDim.y;
>	if (Col < width && Row < height) {
>		// get 1D coordinate for the greyscale image linearized
>		int greyOffset = Row*width + Col;
>		// one can think of the RGB image haveing CHANNEL times
>		// comumns than the grayscale image (rgb=3)
>		int rgbOffset = greyOffset*CHANNELS;
>		unsigned char r = Pin[rgbOffset];   // red value for pixel
>		unsigned char g = Pin[rgbOffset+1]; // green value for pixel
>		unsigned char b = Pin[rgbOffset+2]; // green value for pixel
>		
>		// perform the rescaling and store it
>		Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b
>	}
>}
>```

>[!example] Image Blur (simplified)
>```c
>__global__ void blurKernel(unsigned char* in, unsigned char* out,
>							int W, int h) {
>	int Col = threadIdx.x + blockIdx.x * blockDim.x;
>	int Row = threadIdx.y + blockIdx.y * blockDim.y;
>	
>	if (Col < width && Row < height) {
>		int pixVal = 0;
>		int pixels = 0;
>		
>		// get the average of the surrounding
>		// BLUR_SIZE x BLUR_SIZE box
>		for(int blur=-BLUR_SIZE; blurRow<BLUR_SIZE+1; ++blurRow) {
>			for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE;++blurCol) {
>				int curRow = Row + blurRow;
>				int curCol = Col + blurCol;
>				// verify we have a valid image pixel
>				if(curRow>-1 && curRow<h && curCol>-1 && curCol<w) {
>					pixVal += in[curRow*w + curCol];
>					pixels++; // keep track of number of pixels in avg
>				}
>			}
>		}
>	}
>	
>	// write our new pixel value out
>	out[Row*w+Col] = (unsigned char)(pixVal/pixels);
>}
>```

---
## Performance estimation

>[!question] How can we measure performance to have an idea of whether we are saturating the computational capabilities of the hardware?
>We use $\text{FLOP/s}$ (i.e., floating-point operations per second)
>
>>[!question] What type of floating-point? 64-bit, 32-bit, 16-bit, …?

Today we have systems capable of 1 ExaFLOP/s (i.e., $10^{18} \text{FLOP/s}$)

We define the **compute-to-global-memory access ratio** as the number of floating-point calculation performed for each access to the global memory within a region of a program (also knwon as arithmetic/operational intensity, measured in $\text{FLOW}/byte$)

>[!example]
>```c
>pixVal += in[cureRow * w + curCol];
>```
>
>All threads access global memory for their input matrix elements. Let’s suppose that the global memory bandwidth is $200\text{ GB/s}$
>
>>[!question] How many operands can we load?
>>$$\frac{200\text{ GB/s}}{4\text{ bytes}}=50G \text{ operands/s}$$
>
>We do one floating-point operation (`+=`) on each operand. Thus, we can expect, in the best case, a peak performance of $50\text{ GFLOP/s}$
>
>Let’s suppose that the peak floating-point rate of this GPU is $1500\text{ GFLOP/s}$
>
>This limits the execution rate to $3.3\%$ of the peak floating-point execution rate of the device (i.e., the memory movement to/from the memory, rather than the compute capacity, is limiting our performance)
>
>To achieve the peak $1.5\text{ TFLOP/s}$ rating of the processor, we need a ration of $30$ or higher (i.e., we would need to perform 30 floating-point operations on every operand)
