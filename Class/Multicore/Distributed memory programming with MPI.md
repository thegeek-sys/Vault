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

```c
// returns total number of processes in the communicator
int MPI_Comm_size(
	MPI_Comm comm,      // in
	int*     comm_sz_p, // out
);

// my rank (the process making this call)
int MPI_Comm_rank(
	MPI_Comm comm,      // in
	int*     my_rank_p, // out
);
```

>[!example]- Hello World! (v1)
>```c
>#include <stdio.h>
>#include <mpi.h>
>
>int main(void) {
>	int comm_sz, my_rank;
>	MPI_Init(NULL, NULL);
>	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
>	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
>	printf(“hello, world from process %d out of %d\n”, my_rank, comm_sz);
>	MPI_Finalize();
>	return 0;
>}
>```
>
>
>```c
>mpiexec -n 4 ./mpi_hello
>// hello, world from process 2 out of 4
>// hello, world from process 3 out of 4
>// hello, world from process 0 out of 4
>// hello, world from process 1 out of 4
>
>mpiexec -n 4 ./mpi_hello
>// hello, world from process 1 out of 4
>// hello, world from process 0 out of 4
>// hello, world from process 3 out of 4
>// hello, world from process 2 out of 4
>```

---
## Communication
### $\verb|MPI_Send|$
```c
int MPI_Send(
	void*        msg_buf_p, // in
	int          msg_size,  // in
	MPI_Datatype msg_type,  // in
	int          dest,      // in
	int          tag,       // in
	MPI_Comm     comm       // in
);
```

Where:
- `msg_buf_p` → buffer that contains the message
- `msg_size` → number of elements of the message, not number of bytes
- `msg_type` → type the message (eg. `char`, `int`, …)
- `dest` → rank of the destination process
- `tag` → used to identify the message
- `comm` → communicator
### $\verb|MPI_Recv|$

```c
int MPI_Recv(
	void*        msg_buf_p, // in
	int          buf_size,  // in
	MPI_Datatype buf_type,  // in
	int          source,    // in
	int          tag,       // in
	MPI_Comm     comm,      // in
	MPI_Status*  status_p   // in
);
```

Where:
- `msg_buf_p` → where we want to store the message
- `buf_size` → number of elements of the buffer, not number of bytes
- `buf_type` → type the message (eg. `char`, `int`, …)
- `source` → rank of the destination process (I could specify from any sender)
- `tag` → used to identify the message
- `comm` → communicator
- `status_p` → informations about what happened during the transmission (eg. who sent the message, …)

