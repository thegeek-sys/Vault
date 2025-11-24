---
Class: "[[Multicore]]"
Related:
---
---
## Parallel $\verb|for|$
Consists of forking a team of threads to execute the following structured block, however the structured block following the parallel for directive must be a for loop.

Furthermore, with the parallel directive the system parallelized the for loop by dividing the iterations of the loop among the threads

>[!example] Trapezoid example
>```c
>h = (b-a/n);
>approx = (f(a) + f(b))/2.0;
>for (i=1; i<n-1; i++)
>	approx += f(a + i*h);
>approx = h*approx;
>```
>
>Becomes:
>```c
>h = (b-a/n);
>approx = (f(a) + f(b))/2.0;
># pragma omp parallel for num_threads(thread_count) reduction(+: approx)
>for (i=1; i<n-1; i++)
>	approx += f(a + i*h);
>approx = h*approx;
>```

>[!info] Legal forms for parallelizable $\verb|for|$ statements
>![[Pasted image 20251124145718.png]]
>
>>[!question] Why?
>>Because it allows the runtime system to determine the number of iterations prior to the execution of the loop

>[!warning] Caveats
>- the variable index must have 

