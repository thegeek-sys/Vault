---
Class: "[[Multicore]]"
Related:
---
---
## DMA
DMA (Direct Memory Access) hardware is a unit specialized to transfer a number of bytes requested by OS between physical memory address space regions (some can be mapped to I/O memory allocations), and is used by `cudaMemcpy()` for better efficiency, as it frees CPU for other tasks and uses system interconnection (e.g. PCIe)

![[Pasted image 20260330125245.png|300]]

---
## Virtual memory management
Modern systems use virtual memory management to map many virtual memory spaces to a single physical memory and virtual addresses (pointer values) are translated into physical addresses

>[!warning] Not all variables and data structures are always in the physical memory
>In fact each virtual address space is divided into pages that are mapped into and out of the physical memory and virtual memory pages can be mapped out of the physical memory (*page-out*) to make room
>
>Whether or not a variable is in the physical memory is checked at address translation time

### Data transfers and virtual memory
DMA uses physical addresses so when `cudaMemcpy()` copies an array, it is implemented as one or more DMA transfers. Address is translated and page presence checked for the entire source and destination regions at the beginning of each DMA transfer.

>[!info]
>No address translation is done for the rest of the same DMA transfer so that high efficiency can be achieved

The OS could accidentally page-out the data that is being read or written by a DMA and page-in another virtual page into the same physical location

To solve this issue come in hand **pinned memory** which are virtual memory pages that are specifically marked so that they cannot be paged-out. To do so they are allocated with a special system API function call.

In this way, CPU memory that serve as the source or destination of a DMA transfer must be allocated as pinned memory.

The DMA used by `cudaMemcpy()` requires that any source or destination in the host memory is allocated as pinned memory. If a source or destination of a `cudaMemcpy()` in the host memory is not allocated in pinned memory, it needs to be first copied to a pinned memory, but this operation can be faster if the host memory source or destination is allocated in pinned memory since no extra copy is needed.

---
## Page-locked memory
Placing program data in page-locked pinned memory saves extra data transfers but can be detrimental for the efficiency of the host’s virtual memory.

Pinned memory can be allocated with `malloc()` followed by a call to `mlock()`. Deallocation is done in the reverse order, i.e. `munlock()` then `free()`. Another option is allocating via `cudaMallocHost()` function.

The performance gain obtained via pinned memory depends on the size of the data to be transferred. The gain can range from $10\%$ to a massive $2.5\times$.

```c
cudaError_t cudaMallocHost(
	void **ptr, // addr of pointer to pinned memory (IN/OUT)
	size_t size // size in bytes of request (IN)
)
```

---
## Bank conflicts in shared memory
Shared memory is split into banks as illustrated below:
![[Pasted image 20260330145310.png]]

>[!info]
>Devices of CC 2.0 and above have 32 banks. Earlier devices had 16.

Each bank can serve one access per cycle (i.e. if threads access different banks in shared memory, access is instantaneous). If threads access different data but on the same bank, the access is serialized.

>[!question] What are Memory Banks?
>Memory banks are parallel memory modules designed to allow multiple threads in a warp to access data simultaneously. In an ideal scenario, if 32 threads in a warp access 32 different banks, the hardware fulfills the request in a single clock cycle.
>
>The data is interleaved across these banks. Successive 32-bit words are assigned to successive banks. You can calculate which bank a specific address belongs to using this formula:
>$$\text{Bank Index}=(\text{Address}/4\text{ bytes}) (\text{mod }32)$$

>[!question] When do we have memory bank conflicts?
>A bank conflict occurs when multiple threads in the same warp request **different** data addresses that happen to reside in the **same memory bank**.
>- the penalty: because a bank can only serve one request per cycle, the hardware must **serialize** the accesses.
>- the result: if 2 threads conflict, it takes 2 cycles; if all 32 threads conflict (e.g., they all access different words in Bank 0), it takes 32 cycles to complete a single memory instruction.

>[!example] 2-way bank conflicts
>Linear addressing `stride==2`. Here, Thread 0 accesses Bank 0, and Thread 16 (in the same warp) also accesses Bank 0 but at a different address.
>This results in a 2-way conflict.
>![[Pasted image 20260330145523.png|300]]

### Bank Broadcast and Multicast
There is one major exception to the conflict rule is the broadcast.

When every thread in a warp accesses the **exact same address** within a bank, the hardware does not serialize the access. Instead, it reads the value once and "broadcasts" it to all requesting threads in a single cycle.
- **broadcast:** all 32 threads read the same word.
- **multicast:** a subset of threads in the warp read the same word.

> [!tip] Accessing the **same** address in a bank is fast (Broadcast). Accessing **different** addresses in the same bank is slow (Conflict).

### Take-home message
Threads in a warp/half-warp should avoid accessing different locations in the same shared memory bank at the same time. To fix conflicts, you can often "pad" your shared memory arrays (e.g., `shared[row][33]` instead of `[row][32]`) to shift the data alignment.