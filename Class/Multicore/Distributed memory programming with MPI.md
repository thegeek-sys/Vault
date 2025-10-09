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
