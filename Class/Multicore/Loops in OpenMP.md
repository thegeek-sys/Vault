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
>
>
>>[!warning] Caveats
>>- the variable `index` must have integer or pointer type (e.g. it can’t be a float)
>>- the expression `start`, `end` and `incr` must have a compatible type. For example, if index is a pointer, then `incr` must have integer type
>>- the expressions `start`, `end`, and `incr` must not change during execution of the loop
>>- during execution of the loop, the variable `index` can only be modified by the “increment expression” in the `for` statement
>>
>>>[!example]
>>>*Cannot* be parallelized
>>>```c
>>>for (i=0; i<n; i++) {
>>>	if (...) break;
>>>}
>>>```
>>>
>>*Cannot* be parallelized
>>>```c
>>>for (i=0; i<n; i++) {
>>>	if (...) return 1;
>>>}
>>>```
>>>
>>>*Can* be parallelized
>>>```c
>>>for (i=0; i<n; i++) {
>>>	if (...) exit();
>>>}
>>>```
>>>
>>>*Cannot* be parallelized
>>>```c
>>>for (i=0; i<n; i++) {
>>>	if (...) i++;
>>>}
>>>```




