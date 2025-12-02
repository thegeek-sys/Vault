---
Class: "[[Multicore]]"
Related:
---
---
## Device properties

>[!example] Listing all the GPUs in a system
>```c
>int deviceCount = 0;
>cudaGetDeviceCount(&deviceCount);
>if (deviceCount == 0)
>	printf("No CUDA compatible GPU exists\n")
>else {
>	cudaDeviceProp pr;
>	for (int i=0; i<deviceCount; i++) {
>		cudaGetDeviceProperties(&pr, i);]
>		printf("Dev #%i is %s\n", i, pr.name);
>	}
>}
>
>struct cudaDeviceProp {
>    char name[256];         // A string identifying the device
>    int major;              // Compute capability major number
>    int minor;              // Compute capability minor number
>    int maxGridSize [3];
>    int maxThreadsDim [3];
>    int maxThreadsPerBlock;
>    int maxThreadsPerMultiProcessor;
>    int multiProcessorCount;
>    int regsPerBlock;       // Number of registers per block
>    size_t sharedMemPerBlock;
>    size_t totalGlobalMem;
>    int warpSize;
>    // ... other fields omitted
>};
>```

---
## Memory hierarchy
### Memory access
Data allocated on host memory is not visible from the GPU, and viceversa

```c
int *mydata = (int *)malloc(sizeof(int)*N);
// populating the array
foo<<<grid, block>>>(mydata, N) // not possible
```

Instead, it must explicitly be copied from/to host to GPU

### Memory allocation and copy
#### Allocate memory on device
```c
cudaError_t cudaMalloc (
    void** devPtr,    // Host pointer address,
                      // where the address of
                      // the allocated device
                      // memory will be stored
    size_t size       // Size in bytes of the
)                     // requested memory block
```

#### Deallocation memory on device
```c
// Frees memory on the device.
cudaError_t cudaFree (
    void* devPtr      // Parameter is the host
)                     // pointer address, returned
                      // by cudaMalloc
```

#### Copy of data from host to device
```c
// Copies data between host and device.
cudaError_t cudaMemcpy (
    void* dst,          // Destination block address
    const void* src,    // Source block address
    size_t count,       // Size in bytes
    cudaMemcpyKind kind // Direction of copy.
)
```

The `cudaMemcpyKind` parameter is also an enumerated type. The `kind` parameter can take one of the following values:
- `cudaMemcpyToHost` (`0`) → host to host
- `cudaMemcpyToDevice` (`1`) → host to device
- `cudaMemcpyDeviceToDevice` (`2`) → device to host
- `cudaMemcpyDeviceToDevice` (`3`) → device to device (for multi-GPU configurations)
- `cudaMemcpyDefault` (`4`) → used when Unified Virtual Address space capability is available

>[!example] Vector addition
>>[!warning] 
>>Do not make any assumption on the order in which threads are executed (depends on how the GPU decides to schedule the blocks)
>
>![[Pasted image 20251201230455.png]]
>
>```c
>// compute vector sum C = A+B, each thread performs one pair-wise addition
>__global__
>void vecAddKernel(float* A, float* b, float* C, int n) {
>	int i = blockDim.x*blockIdx.x+threadIdx.x;
>	// if needed because we might have more threads than elements 
>	// in the array (if the number of elements is not a multiple
>	// of block size)
>	if (i<n) C[i] = A[i]+B[i];
>}
>
>// h_ to indicate data allocated on host memory
>void vecAdd (float* h_A, float* h_B, float* h_C, int n) {
>	int size = n*sizeof(float);
>	// d_ to indicate data allocated on device (GPU) memory
>	float *d_A, *d_B, *d_C;
>	
>	cudaMalloc((void **) &d_A, size);
>	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
>	cudaMalloc((void **) &d_B, size);
>	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
>	
>	cudaMalloc((void **) &d_C, size);
>	
>	// each block has 256 threads, and we have n/256 blocks
>	vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
>	
>	cudaMemcpy(d_C, h_C, size, cudaMemcpyDeviceToHost);
>	
>	// free device memory for A, B, C
>	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
>}
>```
>
>![[Pasted image 20251201232233.png]]
>
>For each `cudaMalloc` we should define an error check like this one:
>```c
>cudaError_t err=cudaMalloc((void **) &d_A, size);
>if (error!=cudaSuccess) {
>	printf("%s in %s at line %d\n", cudaGetErrorString(err).__FILE__.__LINE__);
>	exit(EXIT_FAILURE)
>}
>```
>
>But there is another (better) way to do it:
>```c
>#define CUDA_CHECK_RETURN(value) {                         \
>    cudaError_t _m_cudaStat = value;                        \
>    if (_m_cudaStat != cudaSuccess) {                       \
>        fprintf(stderr, "Error %s at line %d in file %s\n", \
>            cudaGetErrorString(_m_cudaStat),                \
>            __LINE__, __FILE__);                            \
>        exit(1);                                            \
>    } \
>}
>
>CUDA_CHECK_RETURN (cudaMalloc (((void **) &da, sizeof(int) * N));
>```

