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
Two kind of components: Master and Workers
Master (one or more) is responsible for:
- handing out pieces of work to workers
- collecting the results of the computations from the workers
- performing I/O duties on behalf of the workers, (ig. sending them the data that they are supposed to process, or accessing a file)
- interacting with the user

It is good for implicit load balancing (no/few inter-worked data exchange) ogni volta che un worker finisce gli viene assegnato altro lavoro
Questa coda comporta che il master è un collo di bottiglia sul master (che divide il lavoro) per questo spesso si ha una gerarchia di master, così da poter evitare di avere un solo punto di fallimento
#### Map-reduce
It’s a variation of master-worker pattern and it’s an old concept, made popular by Google’s search engine

  il tipo di operazione è molto specifico map o reduce
  - map → apply a function on data, resulting in a set of partial results
  - reduce → collect the partial results and derive the complete one

Map and reduce workers can vary in number

>[!info] Master-worker vs. map-reduce
>- master-worker → same function applied to different data items
>- map-reduce → same function applied to different parts of a single data item (data parallel)
