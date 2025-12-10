---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
The **cache** is collection of memory locations that can be accessed in less time than some other memory locations. A CPU cache is typically located on the same chip, or one that can be accessed much faster than ordinary memory (it is usually physically closer)

Also uses more performing (but also more expensive) technology (e.g. SRAM instead of DRAM), so it is going to be faster but smaller

### What to cache
We assume **locality** (of both for instructions and data, i.e., accessing one location is followed by an access of a nearby locations).

There are two kinds of locality:
- *spatial locality* → accessing a nearby location
- *temporal locality* → accessing it in the near future

>[!example] Locality
>```c
>// z[i] follows z[i-1] in memory (spatial locality)
>float z[1000];
>...
>sum = 0.0;
>// access to z[i] follows access to z[i-1] (temporal locality)
>for (i=0; i<1000; i++)
>	z += z[i];
>```

### Cache lines
Data is transferred from memory to cache in blocks/lines (i.e., when `z[0]` is transferred from memory to cache also `z[1]`, `z[2]`, …, `z[15]` might be transferred), in fact doing one transfer of 16 memory locations, is better than doing 16 transfers of one memory location  each

For this reason when accessing `z[0]` you need to wait for the transfer, but then you will find the other 15 elements in cache already

### Cache levels
![[Pasted image 20251209172050.png]]

Data stored in L1 might or might not be stored in L2/L3 as well (it depends on the type of the cache). The CPU first checks if the data is in L1, if not, checks in L2, etc.

>[!question] Why do we care?
>To write efficient/performant parallel code:
>- its sequential parts must be efficient/performant (try to think about how your application accesses the data, random accesses are much worst than linear accesses)
>- the coordination between these sequential parts must be done efficiently

### Consistency
Let’s suppose `x` in cache. If we update the value of `x` in cache, the copy of `x` in main memory is not updated so we have two different values of `x` in main memory and in cache

![[Pasted image 20251210132112.png]]

So when a CPU writes data to cache, the value in cache may be inconsistent with the value in main memory. This problem can be solved in two ways:
- write-through → caches handle this by updating the data in main memory at the time it is written to cache
- write-back → caches mark data in the cache as dirty, and when the cache line is replaced by a new cache line from memory, the dirty line is written to memory

---
## Caching on multicores
![[Pasted image 20251210132757.png|400]]