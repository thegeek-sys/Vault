---
Class: "[[Multicore]]"
Related:
---
---
## Index
- [[#Globally Parallel, Locally Sequential|Globally Parallel, Locally Sequential]]
	- [[#Globally Parallel, Locally Sequential#Single-Program, Multiple Data|Single-Program, Multiple Data]]
	- [[#Globally Parallel, Locally Sequential#Multiple-Program, Multiple Data|Multiple-Program, Multiple Data]]
	- [[#Globally Parallel, Locally Sequential#Master-Worker|Master-Worker]]
	- [[#Globally Parallel, Locally Sequential#Map-reduce|Map-reduce]]
- [[#Globally Sequential, Locally Parallel|Globally Sequential, Locally Parallel]]
	- [[#Globally Sequential, Locally Parallel#Fork/join|Fork/join]]
	- [[#Globally Sequential, Locally Parallel#Loop parallelism|Loop parallelism]]
- [[#Example: trapezoidal rule in MPI|Example: trapezoidal rule in MPI]]
---
## Introduction
We can distinguish the parallel program structure patterns into two major categories:
- **Globally Parallel, Locally Sequential** (*GPLS*) → this means that the application is able to perform multiple tasks concurrently, with each task running sequentially. Patterns that fall in this category include:
	- Single-Program, Multiple Data
	- Multiple-Program, Multiple Data
	- Master-Worker → one master process coordinates and divides the data that each worker has to compute
	- Map-reduce → fase map e fase di reduce, utilizzato su stream di array (supponiamo di voler calcolare  la somma tra tutti gli elementi dell’array, la fase di  map divide l’array ai vari processi e il reduce è l’operazione di somma eseguita dai singoli processi) has a map phase and a reduce phase, it’s used on arrays (calculate sum of the elements of an array)
- **Globally Sequential, Locally Parallel** (*GSLP*) → this means that the application executes as a sequential program, with individual parts of it running in parallel when requested.  Patterns that fall in this category include:
	- fork/join
	- loop parallelism → if it’s possible, execute the loop in parallel

![[Pasted image 20251021173645.png]]

---
##  Globally Parallel, Locally Sequential
### Single-Program, Multiple Data
Keeps all the application logica in a single program
Typical program structure involves:
- program initialization (eg. runtime initialization)
- obtaining a unique identifier → identifiers are numbered from $0$, enumerating the threads or process used. Some systems use vector identifiers (eg. CUDA)
- running the program → execution path diversified based on ID
- shutting down the program → clean-up, saving results, etc.

### Multiple-Program, Multiple Data
SPMD fails when:
- memory requirements are too high for all nodes (in SPMD different processes can’t work on shared memory)
- heterogenous platforms are involved (the same SPMD can’t run both on CPU and GPU)

The execution steps are identical as SMPD but deployment involves different programs

>[!tip] Both SMPD and MPMD are supported by MPI

### Master-Worker
It consists of two components: master and worker
Master (one or more) is responsible for:
- handing out pieces of work to workers
- collecting the results of the computations from the workers
- performing I/O duties on behalf of the workers (ig. sending them the data that they are supposed to process, or accessing a file)
- interacting with the user

It is good for implicit load balancing (no/few inter-worked data exchange), in fact every time that a worker finishes to compute it gets another task until there are no more tasks available
But for the same reason the master could be a bottle neck, so often there is a hierarchy of masters (more points of failure)

### Map-reduce
It’s a variation of master-worker pattern and it’s an old concept, made popular by Google’s search engine

  il tipo di operazione è molto specifico map o reduce
  - map → apply a function on data, resulting in a set of partial results
  - reduce → collect the partial results and derive the complete one

Map and reduce workers can vary in number

>[!info] Master-worker vs. map-reduce
>- master-worker → same function applied to different data items
>- map-reduce → same function applied to different parts of a single data item (data parallel)

---
## Globally Sequential, Locally Parallel
### Fork/join
Single parent thread of execution and the children are created dynamically at run-time. Tasks may run via spawning of threads, or via use of a static pool of threads (creating/destroying processes can be slow, so usually the processes are set idle until they are used again).

Children tasks have to finish for parent thread to continue

>[!tip] Used by OpenMP and Pthread

>[!example]
>```c
>mergesort(A, lo, hi):
>	if lo < hi:                    // at least one element of input
>		mid = floor(lo+(hi-lo)/2)
>		fork mergesort(A, lo, mid) // process (potentially) in parallel
>								   // parallel with main task
>		mergesort(A, mid, hi)      // main task handles second recursion
>		join
>		merges(A, lo, mid, hi)
>```

### Loop parallelism
Employed for migration of legacy/sequential software to multicore. Focuses on breaking up loops by manipulating the loop control variable (but a loop has to be in a particular form to support this, eg. if there is a loop that executes 10 times, 10 threads are executed and each one compute one cycle)

It has limited flexibility, but limited development effort as well

>[!tip] Supported by OpenMP

---
## Example: trapezoidal rule in MPI

>[!example] Trapezoidal rule in MPI
>![[Pasted image 20251024180715.png]]
>
>![[Pasted image 20251024180827.png|300]]
>$$\text{Area of one trapezoid}=\frac{h}{2}[f(x_{i})+f(x_{i+1})]$$
>$$x_{0}=a,\;x_{1}=a+h,\;x_{2}=a+2h, \dots,x_{n-1}=a+(n+1)h,\;x_{n}=b$$
>$$h=\frac{b-a}{n}$$
>$$\begin{align}&\text{Sum of trapezoids}=\sum_{i=0}^{n-1} \frac{h}{2}[f(x_{i})+f(x_{i+1})]= \\&= \frac{h}{2}[f(x_{0})+f(x_{1})+f(x_{1})+f(x_{2})+\dots+f(x_{n-1})+f(x_{n-1})+f(x_{n})]= \\&=h\left[ \frac{f(x_{0})}{2} +f(x_{1})+f(x_{2})+\dots+f(x_{n-1})+\frac{f(x_{n})}{2}\right]\end{align}$$
>
>*Pseudo-code for a serial program*
>```c
>// input: a, b, n
>h = (b-a)/n;
>approx = (f(a) + f(b))/2.0;
>for (i=1; i<=n-1; i++) {
>	x_i = a + i*h;
>	approx += f(x_i);
>}
>approx = h*approx
>```
>
>*Parallelizing the Trapezoidal Rule*
>1. partition problem solution into tasks
>2. identify communication channels between tasks
>3. aggregate tasks into composite tasks
>4. map composite tasks to cores
>
>![[Pasted image 20251025223843.png|430]]
>
>Here it is the pseudo-code
>
>```c
>get a, b, n;
>h = (b-a)/n
>local_n = n/comm_z // number of trapezoid for each core
>local_a = a + my_rank*local_n*h; // from where the core has to start
>local_b = local_a + local_n*h; // where the core has to end
>local_integral = Trap(local_a, local_b, local_n, h);
>
>if (my_rank != 0) {
>	Send local_integral to process 0;
>} else {
>	total_integral = local_intergral;
>	for (proc=1; proc<comm_sz; proc++) {
>		Receive local_integral from proc;
>		total_integral += local_integral;
>	}
>}
>
>if (my_rank == 0) {
>	print result;
>}
>```
>
>Now let’s implement it
>>[!done]- Implementation
>>```c
>>int main(void) {
>>	int my_rank, comm_sz, n=1024, local_n;
>>	double a=0.0, b=3.0, h, local_a, local_b;
>>	double local_int, total_int;
>>	int source;
>>	
>>	MPI_Init(NULL, NULL);
>>	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
>>	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
>>	
>>	h = (b-a)/n;        // h is the same for all the processes
>>	local_n = n/comm_sz // so is the number of trapezoid
>>	
>>	local_a = a + my_rank*local_n*h;
>>	local_a + local_n*h;
>>	local_int = Trap(local_a, local_b, local_n, h);
>>	
>>	if (my_rank != 0) {
>>		MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
>>	} else {
>>		total_int = local_int;
>>		for (source=1; source<comm_sz; source++) {
>>			MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
>>			total_int += local_int;
>>		}
>>	}
>>	
>>	if (my_rank == 0) {
>>		printf("With n = %d trapezoids, our estimate\n", n);
>>		printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
>>	}
>>	MPI_Finalize();
>>	return 0;
>>}
>>
>>double Trap(
>>			double left_endpt, double right_endpt,
>>			int trap_count, double base_len) {
>>	double estimate, x;
>>	int i;
>>	
>>	estimate = (f(left_endpt) + (right_endpt))/2.0;
>>	for (i=1; i<=trap_count-1; i++) {
>>		x = left_endpt + i*base_len;
>>		estimate += f(x);
>>	}
>>	estimate = estimate*base_len;
>>	
>>	return estimate;
>>}
>>```
>
>>[!bug] Issues with the trapezoidal rule implementation
>>When doing the global sum, $p-1$ processes send their data to one process, which then computes all the sums. How long does it take?
>>- for process $0$ → $(p-1)\cdot(T_{\text{sum}}+T_{\text{recv}})$
>>- for all the other processes → $T_{\text{send}}$
>>
>>An alternative is to use a computation tree like this one:
>>![[Pasted image 20251025230745.png|350]]
>> that would result for process $0$ → $\log_{2}(p)\cdot(T_{\text{sum}}+T_{\text{recv}})$
>>
>>The optimal way to compute a global sum depends on the number of processes, the size of the data, and the system we are running on. So having a native way to express would simplify programming and improve performance, for this reason operations like [[Collective operations#$ verb MPI_Reduce $|MPI_Reduce]] exist

