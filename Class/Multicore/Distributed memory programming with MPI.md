---
Created: 2025-10-09
Class: "[[Multicore]]"
Related:
---
---
## Distributed memory systems
![[Pasted image 20251009165706.png]]

In this case to write data from one memory to another one, the CPU has to send a message to another CPU where it says to write in memory

---
## Single-Program Multiple Data
For SPMD we compile one program, and the same program is executed by multiple processes but they do not share memory, so communications happen through **message passing**

You have to use if-else statement to specify what each process must do (similar to what happens when you fork a process)

```c
#include <stdio.h>
#include <mpi.h>

int main(void) {
	MPI_Init(NULL, NULL); // opening
	printf("hello world\n");
	MPI_Finalize();       // closing
	return 0;
}
```

---
## MPI programs
MPI programs are written in C and has uses `mpi.h` header file

>[!info] Notation
>Identifiers defined by MPI start with `MPI_` and the first letter after the underscore is uppercase

### MPI components
#### $\verb|MPI_Init|$
Tells MPI to do all the necessary setup

```c
int MPI_Init(
	int*    argc_p, // in/out
	char*** argv_p  // in/out
);
```

#### $\verb|MPI_Finalize|$
Tells MPI we’re done, so clean up anything allocated for this program

```c
int MPI_Finalize(void);
```

### Basic outline

```c
...
#include <mpi.h>
...
int main(int argc, char* argv[]) {
	...
	// no MPI calls before this
	MPI_Init(&argc, &argv);
	...
	MPI_Finalize();
	// no MPI calls after this
	...
	return 0;
}
```

---
## Compilation

```c
mpicc -g -Wall -o mpi_hello mpi_hello.c
```

Where:
- `mpicc` → wrapper script to compile
- `-g` → produce debugging information
- `-Wall` → turns on all warnings

---
## Execution

```c
mpiexec -n <number of processes> <executable>
```

>[!example]- Example
>```c
>mpiexec -n 1 ./mpi_hello
>// hello, world
>
>mpiexec -n 4 ./mpi_hello
>// hello, world
>// hello, world
>// hello, world
>// hello, world
>```

---
## Debugging
Parallel debugging is trickier than debugging serial programs. In fact many processes are computing so getting the state of one failed process is usually hard

With MPI we can use ddd (or gdb) on one process:
```c
mpiexec -n 4 ./test : -n 1 ddd ./test : -n 1 ./test
// launches the 5th process under "ddd" and all other processes normally
```

---
## Identifying MPI process
Common practice to identify processes is with nonnegative integer called **ranks**. The $p$ processes are numbered $0,1,\dots,p-1$

### Communicators
Communicators are a collection of processes that can send messages to each other

`MPI_Init` defines a communicator that consists of all the processes created when the program is started called **`MPI_COMM_WORLD`**

