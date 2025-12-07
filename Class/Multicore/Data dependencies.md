---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
A data dependency occurs when a program instruction depends from the result of another previous instruction. In a parallel context, this it critical, because it breaks the logical chain: if two operations have to occur in a specific sequence, they can’t be executed simultaneously by different threads without precautions.

Let’s analyze this example:
```c
fibo[0] = fibo[1] = 1;

#pragma omp parallel for num_threads(2)
for (i = 2; i < n; i++)
	fibo[ i ] = fibo[ i – 1 ] + fibo[ i – 2 ];
```

Usually we get as output `1 1 2 3 5 8 13 21 34 55` which is correct, but sometimes we can get something like `1 1 2 3 5 8 0 0 0 0`

>[!question] What happened?
>OpenMP compilers don’t check 