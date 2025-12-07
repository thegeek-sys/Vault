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