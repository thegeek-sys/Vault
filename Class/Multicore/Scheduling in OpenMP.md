---
Class: "[[Multicore]]"
Related:
---
---
## Loops
### Parallel $\verb|for|$
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

### Nested $\verb|for|$ loops

>[!info]- Possible solutions
>If we have nested `for` loops, it is often enough to simply parallelize the outermost loop:
>```c
>a();
># pragms omp parallel for
>for (int i=0; i<4; ++i) {
>	for (int j=0; j<4; ++j) {
>		c(i, j);
>	}
>}
>z();
>```
>![[Pasted image 20251124223424.png|350]]
>
>
>But sometimes the outermost loop is so short that not all threads are utilized:
>```c
>a();
>// 3 iterations, so it won't have send to start more than 3 threads
># pragms omp parallel for
>for (int i=0; i<3; ++i) {
>	for (int j=0; j<6; ++j) {
>		c(i, j);
>	}
>}
>z();
>```
>![[Pasted image 20251124223654.png|350]]
>
We could try to parallelize the inner loop, but there is no guarantee that the thread utilization is better:
>```c
>a();
>for (int i=0; i<3; ++i) {
>	# pragms omp parallel for
>	for (int j=0; j<6; ++j) {
>		c(i, j);
>	}
>}
>z();
>```
>![[Pasted image 20251124223821.png|350]]

The **correct solution** is to **collapse it into one loop** that does 18 iterations. We can do it manually:
```c
a();
#pragma omp parallel for
for (int ij = 0; ij < 3*6; ++ij) {
	c(ij / 6, ij % 6);
}
z();
```
![[Pasted image 20251124224029.png|350]]

But it can also be automated by using OpenMP:
```c
a();
#pragma omp parallel for collapse(2)
for (int i=0; i<3; ++i) {
	# pragms omp parallel for
	for (int j=0; j<6; ++j) {
		c(i, j);
	}
}
z();
```

>[!error] Wrong way
>”Nested parallelism” is disabled in OpenMP by default (i.e. inner parallel pragmas will be ignored)
>```c
>a();
># pragma omp parallel for
>for (int i=0; i<3; ++i) {
>	# pragma omp parallel for
>	for (int j=0; j<6; ++j) {
>		c(i, j);
>	}
>}
>z();
>```
>
>![[Pasted image 20251124224518.png|350]]
>
>If “Nested parallelism” is enabled it will create 12 threads on a server with 4 cores ($3\cdot 4$)!

---
## Scheduling loops

```c
schedule(type, chunksize)
```

type` can be:
- `static` → the iterations can be assigned to the threads before the loop is executed
- `dynamic` or `guided` → the iterations are assigned to the threads while the loop is executing
- `auto` → the compiler and/or the run-time system determine the schedule
- `runtime` → the schedule is determined at run-time

The `chunksize` is a positive integer and can be omitted. When it’s omitted, a `chunksize` of 1 is used

>[!example]-
>We want to parallelize this loop
>```c
>// calls the sin function i times
>double f(int i) {
>	int j, start = i*(i+1)/2, finish = start + i;
>	double return_val = 0.0;
>	
>	for (j=start; j<=finish; j++) {
>		return_val += sin(j);
>	}
>	return return_val;
>}
>
>sum = 0.0
>for (i=0; i<=n; i++)
>	sum += f(i)
>```
>
>>[!question] In practice, how are iterations assigned to threads?
>>Default partitioning
>>![[Pasted image 20251203122134.png]]
>>
>>Cyclic partitioning
>>![[Pasted image 20251203122230.png]]
>
>Results
>![[Pasted image 20251203122816.png]]
>
>##### Default schedule
>```c
>sum = 0.0
>#pragma omp parallel for num_threads(thread_count) \
>	reduction(+:sum)
>for (i=0; i<=n; i++)
>	sum += f(i)
>```
>
>##### Cyclic schedule
>```c
>sum = 0.0
>#pragma omp parallel for num_threads(thread_count) \
>	reduction(+:sum) schedule(static, 1)
>for (i=0; i<=n; i++)
>	sum += f(i)
>```

### $\verb|static|$
With `static`, OpenMP divides the iteration range ahead of time into contiguous block of size `chunksize`. So each thread receives a chunk of `chunksize` iterations

>[!example]
>Twelve iterations, $0,1,\dots,11$, and three threads
>```c
>schedule(static,1)
>```
>$$\begin{align*} \text{Thread 0: } &\quad 0, 3, 6, 9 \\ \text{Thread 1: } &\quad 1, 4, 7, 10 \\ \text{Thread 2: } &\quad 2, 5, 8, 11 \end{align*} $$
>
>```c
>schedule(static,2)
>```
>$$\begin{align*} \text{Thread 0: } &\quad 0, 1, 6, 7 \\ \text{Thread 1: } &\quad 2, 3, 8, 9 \\ \text{Thread 2: } &\quad 4, 5, 10, 11 \end{align*} $$
>
>```c
>schedule(static,4)
>```
>$$\begin{align*} \text{Thread 0: } &\quad 0, 1, 2, 3 \\ \text{Thread 1: } &\quad 4, 5, 6, 7 \\ \text{Thread 2: } &\quad 8, 9, 10, 11 \end{align*} $$

### $\verb|dyanamic|$ or $\verb|guided|$
The iterations are also broken up into chunks of `chunksize` consecutive iterations. Each thread executes a chunk, and when a thread finishes a chunk, it requests another one from the run-time system and this continues until all iterations are completed.

In this case we have better load balancing, but higher overhead to schedule the chunks (can be tuned through the `chunksize`)
