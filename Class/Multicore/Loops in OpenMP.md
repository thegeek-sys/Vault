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

>[!example] Odd-even sort
>In this case the `pragma` directive might fork/join new threads every time it is called (depends on the implementation). If it does so, we would have some overhead
>```c
>for (phase=0; phase<n; phase++) {
>	if (phase%2 == 0) {
>		# pragma omp parallel for num_theads(thread_count) \
>		default(none) shared(a, n) private(i, tmp)
>		for (i=1; i<n; i+=2) {
>			if (a[i-1] > a[i]) {
>				tmp = a[i-1];
>				a[i-1] = a[i];
>				a[i] = tmp;
>			}
>		}
>	} else {
>		# pragma omp parallel for num_threads(thread_count) \
>		default(none) shared(a, n) private(i, tmp)
>		for (i=1; i<n-1; i+=2) {
>			if (a[i] > a[i+1]) {
>				tmp = a[i+1];
>				a[i+1] = a[i];
>				a[i] = tmp;
>			}
>		}
>	}
>}
>```
>
>Is it possible to create the threads at the beginning (before line 1)?
>>[!done] Solution
>>```c
>># pragma omp parallel num_theads(thread_count) \
>>default(none) shared(a, n) private(i, tmp, phase)
>>for (phase=0; phase<n; phase++) {
>>	if (phase%2 == 0) {
>>		# pragma omp for
>>		for (i=1; i<n; i+=2) {
>>			if (a[i-1] > a[i]) {
>>				tmp = a[i-1];
>>				a[i-1] = a[i];
>>				a[i] = tmp;
>>			}
>>		}
>>	} else {
>>		# pragma omp for
>>		for (i=1; i<n-1; i+=2) {
>>			if (a[i] > a[i+1]) {
>>				tmp = a[i+1];
>>				a[i+1] = a[i];
>>				a[i] = tmp;
>>			}
>>		}
>>	}
>>}
>>```
>>
>>reusing the same threads provide faster execution times
>>![[Pasted image 20251124215509.png|350]]
