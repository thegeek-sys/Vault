---
Class: "[[Multicore]]"
Related:
---
---
## Master/single directives
Both `master` and `single` directive force the execution of the following structured block by a single thread.

There is a significant difference: single implies a *barrier* on exit from the block, even tho there other differences like with `master` the block is guaranteed to be executed by the master thread

```c
int examined = 0;
int prevReported = 0;
#pragma omp for shared(examined, prevReported)
for (int i=0; i<N; i++) {
	// some processing
	
	// update the counter
	#pragma omp atomic
	examined++;
	
	// use the master to output an update every 1000 newly
	// finished iterations
	#pragma omp master 
	{
		int temp = examined;
		if (temp-prevReported >= 1000) {
			prevReported = temp;
			printf("Examined %.21f%%\n", temp*1.0/N);
		}
	}
}
```

---
## Barrier directive
The `barrier` directive block until all team threads reach that point

```c
int main() {
	int a[5], i;
	#pragma omp parallel
	{
		// perform some computation
		#pragma omp for
		for (i=0; i<5; i++)
			a[i] = i*i;
		
		// print immediate result
		#pragma omp master
		for (i=0; i<5; i++)
			printf("a[%d] = %d\n", i, a[i]);
		
		// wait
		#pragma omp barrier
		
		// continue with the computation
		#pragma omp for
		for (i=0; i<5; i++)
			a[i] += i;
	}
}
```

---
## The section/sections directives
How can we send different task in parallel?

```c
#pragma omp parallel
switch (omp_get_thread_num()) {
	case 0: {
		//concurrent block 0
	}
	break;
	case 1: {
		//concurrent block 1
	}
	break;
}
```

>[!warning] Why this option is not recommended
>- **Lack of Correctness (Safety Issue)** → if the program runs with fewer threads than the number of `case` statements (e.g., on a single-core machine), the code inside the higher `case` blocks will never be executed, leading to incorrect results.
>- **No Automatic Scalability** → you have to manually calculate how to distribute M tasks across N threads. If the number of available threads changes, you might need to rewrite the code to ensure all tasks are covered.
>- **Poor Load Balancing** → it enforces a "static assignment". If Thread 0 finishes a short task quickly while Thread 1 is stuck on a long task, Thread 0 cannot help with the remaining work; it just sits idle.
>- **Rigidity and Maintainability** → hard-coding thread IDs makes the code rigid and harder to read. Using `omp sections` delegates the complexity of task scheduling and thread management to the OpenMP runtime, which is safer and cleaner.

The individual work items are contained in blocks decorated by section directives:
```c
#pragma omp parallel sections
{
    #pragma omp section
    {
        // concurrent block 0
    }
    // ...
    #pragma omp section
    {
        // concurrent block M-1
    }
}
```

`omp parallel sections` directive combines the `omp parallel` and `omp sections` directives. There is an implicit barrier at the end of a `sections` construct unless a `nowait` clause is specified

---
## Ordered construct
The construct `ordered` is used inside a parallel for, to ensure that a block will be executed as if in sequential order.

```c
double data[N];
#pragma omp parallel shared(data, N)
{
    #pragma omp for ordered schedule(static, 1)
    for(int i = 0; i < N; i++)
    {
        // process the data
		
        // print the results in order
        #pragma omp ordered
        cout << data[i];
    }
}
```