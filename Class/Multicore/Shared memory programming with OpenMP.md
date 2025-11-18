---
Class: "[[Multicore]]"
Related:
---
---
## OpenMP
OpenMP (multi processing) is an API for shared-memory parallel programming, and is designed for shared-memory systems. System is viewed as a collection of cores or CPUs, all of which have access to main memory.

>[!warning]
>Not to be confused with OpenMPI

OpenMP aims to **decompose a sequential program into components** that can be executed in parallel and allows an “incremental” conversion of sequential programs into parallel ones, with the assistance of the compiler (much less invasive than MPI). OpenMP relies on compiler directives for decorating portions of the code that the compiler will attempt to parallelize.

OpenMP programs are globally sequential, locally parallel and they follow the fork-join paradigm:
![[Pasted image 20251111172158.png|400]]

---
## Pragmas
Pragmas are special preprocessor instructions typically added to a system to allow behaviors that aren’t part of the basic C specification.

>[!info]
>If the compiler doesn’t support the pragmas, it ignores them.

```c
#pragma
```

### OpenMP pragmas
