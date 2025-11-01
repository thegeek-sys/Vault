---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
A process is an instance of a running (or suspended) program, while threads are analogous to a “light-weight” process. In a shared memory program a single process may have multiple threads of control

---
## OpenMP vs. Pthreads
OpenMP provides a higher abstraction interface. On POSIX systems, it is most likely implemented on top of Pthreads, while on non-POSIX systems (e.g. Windows), it will be implemented on top of some other threading abstraction. Thus, it is more portable than Pthreads (you write the OpenMP code and run it “everywhere“ without modifying it).
Some OpenMP construct can also be used to run code on GPUs.

Pthreads provides you with much more fine-grained control (as usual is a matter of trade-offs between ease-of-use and flexibility/performance).

Mixing OpenMP and Pthreads in the same code should work, but there might be corner cases where it does not. It is usually better to use either one or the other unless you have very string reasons to mix them.

---
## How many threads should we run?
In principle, you should try to avoid having more threads than cores but there are situations where it might make send.

>[!question] How to check how many cores do you have?
>```bash
>$ lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('
>```

---
## Threads safety in MPI
To avoid problems of shared memory in MPI, to make it thread safe you have to initialize it with the following command

```c
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
```

Where:
- `argc`, `argv` → same as `MPI_Init`
- `required` → “threading level” (in)
- `provided` → supported “threading level” (out)

### Threading levels in MPI
`MPI_THREAD_SINGLE` → ranks is not allowed to use threads, which is basically equivalent to calling MPI_Init
`MPI_THREAD_FUNNELED` → MPI rank can be multi-threaded but only the main thread may call MPI functions. Ideal for fork-join parallelism such as used in `#pragma` omp parallel, where all MPI calls are outside the OpenMP regions
`MPI_THREAD_SERIALIZED` → rank can be multi-threaded but only one thread at a time may call MPI functions. The rank must ensure that MPI is used in a thread-safe way. One approach is to ensure that MPI usage is mutally excluded by all the threads
`MPI_THREAD_MULTIPLE` → rank can be multi-threaded and any thread may call MPI functions. The MPI library ensures that this access is safe  across threads. Note that this makes all MPI operations less efficient, even if only one thread makes MPI calls, so should be used only when necessary

>[!warning]
>Not all the threading levels are supported by all the MPI implementations (e.g. some implementations might not support `MPI_THREAD_MULTIPLE`)

>[!example] `MPI_THREAD_SINGLE`
>![[Pasted image 20251101221403.png]]

>[!example] `MPI_THREAD_FUNNELED`
>![[Pasted image 20251101221442.png]]

>[!example] `MPI_THREAD_SERIALIZED`
>![[Pasted image 20251101221519.png]]

>[!example] `MPI_THREAD_MULTIPLE`
>![[Pasted image 20251101221600.png]]

