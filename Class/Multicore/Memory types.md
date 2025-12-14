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