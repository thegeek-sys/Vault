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

