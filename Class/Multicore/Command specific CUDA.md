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