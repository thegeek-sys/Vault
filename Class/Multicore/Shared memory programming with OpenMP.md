---
Class: "[[Multicore]]"
Related:
---
---
## OpenMP
OpenMP (multi processing) is an API for shared-memory parallel programming, and is designed for shared-memory systems. System is viewed as a collection of cores or CPUs, all of which have access to main memory.

>[!warning]
>Not to be confused with OpenMPI

OpenMP aims to **decompose a sequential program into components** that can be executed in parallel and allows an “incremental” conversion of sequential programs into parallel ones, with the assistance of the compiler (much less invasive than MPI). OpenMP relies on compiler directives for decorating portions of the code that the compiler will attempt to parallelize.

>[!info] Some teminology
>- in OpenMP parlance the collection of threads executing the parallel block, the original thread and the new threads, is called a team
>- master → the original thread of execution
>- parent → thread that encountered a parallel directive and started a team of threads.
>- in many cases, the parent is also the master thread.
>- child → each thread started by the parent is considered a child thread.

OpenMP programs are globally sequential, locally parallel and they follow the fork-join paradigm:
![[Pasted image 20251111172158.png|400]]

---
## Pragmas
Pragmas are special preprocessor instructions typically added to a system to allow behaviors that aren’t part of the basic C specification.

>[!info]
>If the compiler doesn’t support the pragmas, it ignores them.

```c
#pragma
```

### OpenMP pragmas

```c
#pragma omp parallel
```

It is the most basic parallel directive, needed to run the following block of code in parallel. The number of threads used is determined by the runtime system.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Hello(void);

int main(int argc, char* argv[]) {
	// get number of threads from command line
	int thread_count = strtol(argv[1], NULL, 10);
	
	#pragma omp parallel num_threads(thread_count)
	Hello();
	
	return 0;
}

void Hello(void) {
	int my_rank = omp_get_thread_num(); // my thread number
	int thread_count = omp_get_num_threads(); // total number of threads
	
	printf("Hello from the thread %d\n", my_rank, thread_count)
}
```

This code snippet runs the `Hello` function with `thread_count` threads. By default `thread_count` is the total number of available cores.

>[!info]
>A OpenMP program waits for all the threads to finish (`join`)
>![[Pasted image 20251111174218.png]]

To run a program of this kind you need to:
```bash
gcc -g -Wall -fopenmp -o omp_hello omp_hello.c # compiling
./omp_hello 4 # running with 4 threads
```

>[!warning]
>The order of execution of the threads is not established

### Thread team size control
It’s possible to change the number of threads in various ways
#### Universally
Via the `OMP_NUM_THREADS` environmental variable

>[!example]
>```c
>$ echo ${OMP_NUM_THREADS} # to query the value
>$ export OMP_NUM_THREADS=4 # to set it in BASH
>```

#### Program level
Via the `omp_set_num_threads` function, outside an OpenMP construct.

#### Pragma level
Via the `num_threads` clause

>[!example]
>If universally is set to $8$ but on pragma to $4$, the remaining $4$ cores remain idle

### clause
A clause is some text that modifies a directive.

>[!example]
>The `num_threads` clause can be added to a parallel directive, it allows the programmer to specify the number of threads that should execute the following block
>
>```c
># pragma omp parallel num_threads(thread_count)
>```

>[!warning]
>- there may be system-defined limitations on the number of threads that a program can start
>- the OpenMP standard doesn’t guarantee that this will actually start `thread_count` threads
>- most current systems can start hundreds or even thousends of threads
>- unless we’re trying to start a lot of threads, we will almost always get the desired number of threads
>- after a block is completed, there is an implicit barrier (`join` is done)

---
## Parallel construct
![[Pasted image 20251111174817.png]]

>[!example]
>If it’s specified $8$ threads just $7$ are created ($8-1$, one of them is the master)

```c
#include <stdio.h>
#include <omp.h>

int main() {
	printf("only master thread here");
	
	#pragma omp parallel
	{
		// each thread has a private copy of a variable
		int tid = omp_get_thread_num();
		printf("Hello I am thread number %d\n", tid);
	}
	
	printf("Only master thread here\n");
}
```

The variables outside the parallel block are shared across all the threads (if a thread modifies it, every thread will see it).

### In case the compiler doesn’t support OpenMP
We use the construct `#ifdef` to make a compiler run certain instructions only if OpenMP is installed and supported

```c
#ifdef _OPENMP
#include <omp.h>
#endif

// ...

# ifdef _OPENMP
int my_rank = omp_get_thread_num ( );
int thread_count = omp_get_num_threads ( );
# else
int my_rank = 0;
int thread_count = 1;
# endif
```

---
## Example: trapezoidal rule in OpenMP

>[!example] Trapezoidal rule
>![[Pasted image 20251111175640.png]]
>
>##### Assignment of trapezoids to threads
>![[Pasted image 20251111175703.png|400]]
>But we get unpredictable results when two (or more) threads attempt to simultaneously execute:
>```c
>global_result += my_result;
>```
>
>![[Pasted image 20251118172506.png|550]]
>
>##### Mutual exclusion
>For this reason we have to introduce mutual exclusion, so that only one thread at a time can execute the structured block that follows `citical` pragma
>```c
># pragma omp citical
>	global_result += my_result;
>```
>
>>[!done]- Solution
>>```c
>>#include <stdio.h>
>>#include <stdlib.h>
>>#include <omp.h>
>>
>>void Trap(double a, double b, int n, double* global_result_p);
>>
>>int main(int argc, char* argvp[]) {
>>	double global_result = 0.0; // store result in global_result
>>	double a, b;                // left and right endpoints
>>	int    n;                   // total number of trapezoids
>>	int    thread_count;
>>	
>>	thread_count = strtol(argv[1], NULL, 10);
>>	printf("Enter a, b, and n\n")
>>	scanf("%lf %lf %d", &a, &b, &n);
>>	
>>	#pragma omp parallel num_threads(thread_count)
>>	Trap(a, b, n, &global_result)
>>	
>>	printf("With n = %d trapezoids, our estimate\n", n);
>>	printf("of the integral from %f to %f =%.14e\n", a, b, global_result);
>>	
>>	return 0;
>>}
>>
>>void Trap(double a, double b, int n, double* global_result_p) {
>>	double h, x, my_result;
>>	double local_a, local_b;
>>	int    i, local_n;
>>	int    my_rank = omp_get_thread_num();
>>	int    thread_count = omp_get_num_threads();
>>	
>>	h = (b - a) / n;
>>	local_n = n / thread_count;
>>	local_a = a + my_rank * local_n * h;
>>	local_b = local_a + local_n * h;
>>	
>>	my_result = (f(local_a) + f(local_b)) / 2.0;
>>	for (i = 1; i <= local_n - 1; i++) {
>>		x = local_a + i * h;
>>		my_result += f(x);
>>	}
>>	
>>	my_result = my_result * h;
>>	
>>	# pragma omp critical
>>	*global_result_p += my_result;
>>}
>>```

---
## Variables scope
In serial programming, the scope of a variable consists of those parts of a program in which the variable can be used

In OpenMP, the scope of a variable refers to the set of threads that can access the variable in a parallel block.

### Scope in OpenMP
