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

---
## Global memory accesses
There are two types of load:
- cached loads
	- used by default for devices that have L1 cache
	- check L1, if not present, check L2, if not present, check global memory
	- load granularity → 128-byte line
- non-cached loads
	- if the device does not have the L1 cache or if has the L1 cache and you compile with `-Xptxas -dlcm=cg`
	- load granularity → 32-byte line

For store L1 is invalidated then write-back to L2 (that’s why we don’t have false sharing)

>[!question] Why should we disable the L1 cache?
>Warp requests 32 aligned, consecutive 4-bytes words (128 bytes)
>![[Pasted image 20260330151512.png]]
>
>Warps requests 32 aligned, permuted 4-byte words (128 bytes)
>![[Pasted image 20260330151556.png]]
>
>Warp requests 32 misaligned, consecutive 4-bytes words (128 bytes)
>![[Pasted image 20260330151630.png]]





