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
- constant memory →