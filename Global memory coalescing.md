---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
In practice, when a thread accesses a memory location, a *burst* of consecutive locations is actually read (similar to when we load a block of consecutive memory locations into a cache line). When all threads in a warp execute a load instruction, the hardware detects if they access consecutive global memory location. In this case, the hardware coalesces all these accesses into a single access (e.g. for a given load instruction of a warp, if thread 0 access global memory location $N$, thread 1 location $N+1$, and so on, all these accesses will be coalesced, or combined into a single request for consecutive locations when accessing the DRAMs).

CUDA devices might impose requirements on the alignment of $N$ (e.g. it must be multiple of 16)

>[!tldr]
>Intuition:  multiple threads in a warp access locations which are close to each other, a single memory transaction is issued, reading a burst of elements
>
>Aligned access: the first address of the transaction is a multiple of the cache granularity (usually, 32 bytes for the L2 cache and 128 bytes for the L1)
>
>Coalesced access: all the 32 threads in a warp access a contiguous memory burst
>