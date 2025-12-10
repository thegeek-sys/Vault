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

### Cache coherence
Programmers have no control over caches and when they get updated

>[!example]
>Let’s imagine this state:
>- `y0` is privately owned by core 0
>- `y1` and `z1` are privately owned by core 1
>- `x=2` is a shared variable
>
>![[Pasted image 20251210133247.png]]
>
>Core 0 cache:
>- `x=7`
>- `y0=2`
>
>Core 1 cache:
>- `x=2`
>- `y1=6`
>- **`z2=4*2 or 4*7?`**
>
>>[!info]
>>Occurs for both write-trough and write-back policies

There are two possible solutions for this problem:
- snooping cache
- directory based cache

#### Snooping cache
The cores share a bus and any signal transmitted on the bus can be “seen” by all cores connected to the bus.

When core 0 updates the copy of `x` stored in its cache it also broadcasts this information across the bus. If core 1 is “snooping” the bus, it will see that `x` has been updated and it can mark its copy of `x` as invalid

>[!warning]
>It’s not used anymore: broadcast is expensive nowadays we have multicores with 64/128 cores

#### Directory based cache
Uses a data structure called a directory that stores the status of each cache line (e.g., a bitmap/list saying which cores has a copy of that line)

When a variable is updated, the directory is consulted, and the cache controllers of the cores that have that variable’s cache line in their caches are invalidated (we can’t invalidate a single variable, but we have to invalidate the whole cache line)

---
## False sharing
Data is fetched for memory to cache in lines. Each line can contain several variables (e.g. if a cache is 64 bytes long, it can contain 16 4-byte integers, which were consecutive in memory). When data is invalidated, the entire line is invalidate.
Even if two threads access two different variables, if those are on the same cache line, this would still cause an invalidation and this would slow down the execution

>[!info] False sharing
>Two threads that writes on different variables on the same cache line. This causes continuous invalidations and heavy execution slow down

We have three possible ways to fix it:
- pad the data
- change the mapping of data to threads/cores
- use private/local variables

### Padding data

>[!done] Solution
>Try to force variables which are accessed by different threads to be on different cache lines.
>
>Padding (but be aware, the compiler might change the order of the fields in a struct). In the following false sharing solution we are assuming 64 bytes cache lines
>```c
>struct alignTo64ByteCacheLine {
>	int _onCacheLine1 __attribute__((aligned(64)))
>	int _onCacheLine2 __attribute__((aligned(64)))
>}
>```
>
>>[!question] How to get the cache line size?
>>- from the code → `sysconf(_SC_LEVEL1_DCACHE_LINESIZE)`
>>- from the shell → `getconf LEVEL1_DCACHE_LINESIZE`
>
>Do all the updates on a variable local to the thread (e.g. allocated on the stack, and then write it on heap only at the end)

>[!warning] Padding caveats
>Padding wastes much memory. Another solution is to make updates on a local variable and doing the write on the heap just at the end

>[!question] What if our data is an array?

Let’s analyze this snippet:
```c
double x[N];
#pragma omp parallel for schedule(static, 1)
for (int i=0; i<N; i++)
	x[i] = someFunc(x[i]);
```

With `schedule(static,1)` we assign one iteration per thread before restarting. Since arrays are allocated contiguously, each 16 elements are on the same cache line (considering 4 bytes per element and 64 bytes cache line)

The padded version would be:
```c
double x[N][8];
#pragma omp parallel for schedue(static, 1)
for (int i=0; i<N; i++)
	x[i][0] = someFunc(x[i][0])
```

In this solution we transform the array into a matrix `8x8` (8 bytes per double), so each element goes on a different cache line. But this solution **can kill cache effectiveness** and **wastes memory**
For this reason, in this case I could have used **data mapping change**

### Data mapping change
Let’s use the same example of before:
```c
double x[N];
#pragma omp parallel for schedule(static, 8)
for (int i=0; i<N; i++)
	x[i] = someFunc(x[i])
```

By changing the chunk size of the `schedule` directive, we have no false sharing since thread 0 accesses 8 consecutive elements that corresponds to the cache line size (64 byte).
But to avoid completely false sharing, the array must start at an address that is a multiple of the cache line size (64 bytes)

>[!example] Pthreads matrix-vector multiplication
>```c
>void *Pth_mat_vect(void* rank) {
>	long my_rank = (long) rank;
>	int i, j;
>	int local_m = m/thread_count;
>	int my_first_row = my_rank*local_m;
>	int my_last_row = (my_rank+1)*local_m - 1;
>	
>	for (i = my_first_row; i <= my_last_row; i++) {
>		y[i] = 0.0;
>	    for (j = 0; j < n; j++)
>		y[i] += A[i][j]*x[j];
>	}
>	
>	return NULL;
>}  /* Pth_mat_vect */
>```
>
>Even tho the following matrix configurations do the same number of operations, they have different execution times