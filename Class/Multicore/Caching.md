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
Data is transferred from memory to cache in blocks/lines. Doing one transfer of 16 memory locations, is better than doing 16 transfers of one memory location  each

When accessing `z[0]` you need to wait for the transfer, but then you will find the other 15 elements in cache already
