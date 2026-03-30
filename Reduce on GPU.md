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