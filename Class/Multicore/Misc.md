---
Class: "[[Multicore]]"
Related:
---
---
## OpenMP + MPI
OpenMP and MPI can be used together to make the most of both of them.

MPI defines 4 levels of thread safety:
- `MPI_THREAD_SINGLE` → one thread exists in program
- `MPI_THREAD_FUNNELED` → only the master thread can make MPI calls. Master is the one that calls `MPI_Init_thread()`
- `MPI_THREAD_SERIALIZED` → multithreaded, but only one thread can make MPI calls at a time
- `MPI_THREAD_MULTIPLE` → multithreaded and any thread can make MPI calls at any time

The safest and easiest method is to use `MPI_THREAD_FUNNELED`:
- it fits nicely with most OpenMP models
- expensive loops are parallelized with OpenMP
- communication and MPI calls can be managed in a safe way between loops just from master thread

```c
$ ./a.out 4
$ Time: 0.40 seconds

$ mpirun –n 1 ./a.out 4
$ Time: 1.17 seconds
```

>[!question] Why?
>Open MPI maps each process on a core. Thus, all the threads created by that process will run on the same core (i.e., 4 threads will run on the same core).
>
>>[!question] How to fix it?
>>```bash
>>$ mpirun --bind-to-none –n 1 ./a.out 4
>>$ Time: 0.40 seconds
>>```
>>
>>>[!question] How to check how Open MPI is binding processes?
>>>```bash
>>>$ mpirun --report-bindings -n 1 ./a.out 4 1024 1024 1024 MCW rank 0 bound to socket 0[core 0[hwt 0-1]]: [BB/../../../../../../..][../../../../../../../..][../../../../../../../..][../../../../../../../..]
>>>```

---
## Profiling
Let’s say I have a piece of code with many functions. How to know where most of the time is spent and, this, where it would make most sense to optimize?

>[!quote]
>Premature optimization is the root of all evil

We could explicitly time each function but it’s boring and time consuming, so we can use a profiler that does that for us [link](https://hpc-wiki.info/hpc/Gprof_Tutorial)

---
## Debugging
Mostly through `gdb`, but `valgrind` is a quite helpful tool as well

