---
Class: "[[Multicore]]"
Related:
---
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

Both SMPD and MPMD are supported by MPI

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
Single parent thread of execution and the children are created dynamically at run-time (can be slow, so usually, instead of creating/destroying, the processes are set idle until they are used again).

Tasks may run via spawning of threads, or via use of a static pool of threads 

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
se ho ciclo da 10 volte eseguo 10 thread nei quali ognuno esegue un ciclo

Employed for migration of legacy/sequential software to multicore. Focuses on breaking up loops by manipulating the loop control variable (but a loop has to be in a particular form to support this)

It has limited flexibility, but limited development effort as well and is supported by OpenMP


local_n numero di trapezi su cui ogni processo deve lavorare
local_a punto da dove devo iniziare a lavorare
local_b punto fino a dove devo lavorare

il processo 0 deve fare la somma


>[!error]
>Non è possibile inviare puntatori tramite MPI (ha senso solo per il processo che invia)
