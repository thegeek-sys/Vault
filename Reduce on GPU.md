---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
![[Pasted image 20260330160941.png]]

### Issues
Some threads perform addition while other do nothing (the odd index ones), so no more than half of threads will be executing at any time

---
## Better way of doing a reduce
![[Pasted image 20260330161218.png]]

>[!tldr]
>Intuition: for the first steps, all the threads in a warp will do the same thing (either sum, or do nothing)
>
>E.g., for a 512 threads block:
>- 1° step: the fist 512 threads read data from the other 512 threads (512 active threads)
>- 2° step: threads `[0, 255]` read data from `[256, 511]`
>- …
>- after the 5° step, the stride will $<32$, so we will have divergence (but only for the last few steps)