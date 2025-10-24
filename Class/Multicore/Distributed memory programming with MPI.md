---
Created: 2025-10-09
Class: "[[Multicore]]"
Related:
---
---
## Index
- [[#Distributed memory systems|Distributed memory systems]]
- [[#Single-Program Multiple Data|Single-Program Multiple Data]]
- [[#MPI programs|MPI programs]]
	- [[#MPI programs#MPI components|MPI components]]
		- [[#MPI components#$\verb|MPI_Init|$|$\verb|MPI_Init|$]]
		- [[#MPI components#$\verb|MPI_Finalize|$|$\verb|MPI_Finalize|$]]
	- [[#MPI programs#Basic outline|Basic outline]]
- [[#Compilation|Compilation]]
- [[#Execution|Execution]]
- [[#Debugging|Debugging]]
- [[#Identifying MPI process|Identifying MPI process]]
	- [[#Identifying MPI process#Communicators|Communicators]]
- [[#Communication|Communication]]
	- [[#Communication#$\verb|MPI_Send|$|$\verb|MPI_Send|$]]
	- [[#Communication#$\verb|MPI_Recv|$|$\verb|MPI_Recv|$]]
		- [[#$\verb|MPI_Recv|$#$\verb|status_p|$ argument|$\verb|status_p|$ argument]]
	- [[#Communication#Sending order|Sending order]]
- [[#Communicators|Communicators]]
	- [[#Communicators#Message matching|Message matching]]
- [[#What happens when you do a $\verb|Send|$|What happens when you do a $\verb|Send|$]]
- [[#Point-to-point communication modes|Point-to-point communication modes]]
- [[#Non-blocking communication|Non-blocking communication]]
	- [[#Non-blocking communication#Non-blocking $\verb|Send|$|Non-blocking $\verb|Send|$]]
	- [[#Non-blocking communication#Non-blocking $\verb|Recv|$|Non-blocking $\verb|Recv|$]]
	- [[#Non-blocking communication#Check for completion|Check for completion]]
	- [[#Non-blocking communication#Summary|Summary]]
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

>[!error] Issues with send and receive
>- exact behavior is determined by the MPI implementation
>- `MPI_Send` may behave differently with regard to buffer size cutoffs and blocking
>- `MPI_Recv` always blocks until a matching message is received
>- even if you know your MPI implementation, stick to what is defined by the standard (don’t assume that the send returns immediately for small buffers), otherwise your code will not be portable

#### $\verb|status_p|$ argument
The `status_p` argument is used to provide informations about the transmission. It contains three attributes: `MPI_SOURCE`, `MPI_TAG`, `MPI_ERROR`

```c
MPI_Recv(recv_buf_p, rect_buf_sz, recv_type, src, recv_tag, recv_comm, &status);
```

It it also possibile to use `MPI_Get_count` to retrieve the number of elements from a receive operation status

```c
int MPI_Get_count(
	MPI_Status*  status_p, // in
	MPI_Datatype type,     // in
	int*         count_p   // out
);
```

### Sending order
MPI requires that messages be nonovertaking. This means that if process $q$ send two messages to process $r$, then the first message sent by $q$ must be available to $r$ before the second message. However, there is no restriction on the arrival of messages sent from different processes

| MPI datatype         | C datatype             |
| -------------------- | ---------------------- |
| `MPI_CHAR`           | `signed char`          |
| `MPI_SHORT`          | `signed short int`     |
| `MPI_LONG`           | `signed long int`      |
| `MPI_LONG_LONG`      | `signed long long int` |
| `MPI_UNSIGNED_CHAR`  | `unsigned char`        |
| `MPI_UNSIGNED_SHORT` | `unsigned short int`   |
| `MPI_UNSIGNED`       | `unsigned int`         |
| `MPI_UNSIGNED_LONG`  | `unsigned long int`    |
| `MPI_FLOAT`          | `float`                |
| `MPI_DOUBLE`         | `double`               |
| `MPI_LONG_DOUBLE`    | `long double`          |
| `MPI_BYTE`           |                        |
| `MPI_PACKED`         |                        |

---
## Communicators
`MPI_Init` defines a communicator called **`MPI_COMM_WORLD`**, but MPI also provides functions to create new communicators

User made communicators could be useful to integrate complex functionalities together

>[!example]
>Suppose you have 2 MPI independent libraries of functions; they don’t communicate with each other, but they do communicate internally
>
>We can do it with tags, assigning tags $[1,n]$ and tags $[n+1,m]$ to the two libraries or **we simply pass one communicator to one library functions and a differente communicator to the other library**

### Message matching
![[Pasted image 20251012165039.png|500]]

Message is successfully received if:
- `recv_type = send_type`
- `recv_buf_sz >= send_buf_sz`

A receiver can get a message without knowing:
- the amount of data in the message
- the sender of the message (`MPI_ANY_SOURCE`)
- or the tag of the message (`MPI_ANY_TAG`)

---
## What happens when you do a $\verb|Send|$
![[Pasted image 20251008145432.png]]

>[!warning]
>When you do a `MPI_Send` you don’t know if the message has been sent or not. Your only guarantee is that when you execute the next instructions the `msg_buf_p` could be edited because MPI have already copied the content in another memory buffer (but definitely the data that has been sent is the one before the edit)

---
## Point-to-point communication modes
`MPI_Send` uses the so called standard communication mode. MPI decides based on the size of the message, whether to block the call until the destination process collects it or to return before a matching receive is issued. The latter is chosen if the message is small enough, making `MPI_Send` locally blocking

There are three additional communication modes:
- **buffered** → in buffered mode the sending operation is always locally blocking (eg. it will return as soon as the message is copied to a buffer). The second difference with the standard communication mode is that the buffer is *user-provided*
- **synchronous** → in synchronous mode, the sending operation will return only after the destination process has initiated and started the retrieval of the message. This is a proper **globally blocking** operation
- **ready** → the send operation will succeed only if a matching receive operation has been initiated already. Otherwise the function returns with an error code. The purpose of this mode is to reduce overhead of handshaking operations

```c
int [MPI_Bsend | MPI_Ssend | MPI_Rsend] (void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```

---
## Non-blocking communication
Buffered sends are considered bad for performance, because the caller has to block, waiting for the copy to take place.

Non-blocking (or immediate) functions, maximize concurrency by returning immediately upon initiating a transfer, allowing communication and computation to overlap.
There are both send and receive immediate variants

>[!example]
>While the copying is being done and/or the NIC is sending/receiving the data, I can compute something else

The downside is that the completion of the operations both end-points, has to be queried explicitly:
- for *senders* so that they can re-use or modify the message buffer
- for *receivers* so that they can extract the message contents

>[!tip]
>Non-blocking communications can be coupled with any communication mode

### Non-blocking $\verb|Send|$

```c
int MPI_Isend(
	void *buf, // addfess of data buffer (IN)
	int count, // number of data items (IN)
	MPI_Datatype datatype, // type of buffer elements (IN)
	int dest, // rank of destination process
	int tag, // label to identify the message (IN)
	MPI_Comm comm, // identifies the communicator
	MPI_Request *req // used to return a handle for checking status (OUT)
)
```

The only differing field from the standard `MPI_Send` is `req`. The `MPI_Request` that is returned, is a handle that allows a query on the status of the operation to take place

With the `Isend` a new thread is opened so you can continue to compute data while the communication is ongoing

### Non-blocking $\verb|Recv|$

```c
int MPI_Isend(
	void *buf, // addfess of data buffer (IN)
	int count, // number of data items (IN)
	MPI_Datatype datatype, // type of buffer elements (IN)
	int source, // rank of destination process
	int tag, // label to identify the message (IN)
	MPI_Comm comm, // identifies the communicator
	MPI_Request *req // used to return a handle for checking status (OUT)
)
```

In the `MPI_Irecv` the `MPI_Status` parameter is replaced by a `MPI_Request` one

### Check for completion
All the non-blocking functions are associated to a wait command. This command could be blocking (returns only when the task is completed) or non-blocking (returns immediately with the state of the task)

**Blocking** (destroys handle)
```c
int MPI_Wait(
	MPI_Request *req, // address of the handle identifying the
					  // operation queried (IN/OUT)
	                  // the call invalidates *req by 
	                  // setting it to MPI_REQUEST_NULL
	MPI_Status *st // addess of the structure that will hold the 
				   // comm. information (OUT)
)
```

**Non-blocking** (destroys handle if operation is successful, `*flag=1`)
```c
int MPI_Test(
	MPI_Request *req, // address of the handle identifying the
					  // operation queried (IN)
	int *flag, // set to true is operation is complete (OUT)
	MPI_Status *st // addess of the structure that will hold the 
				   // comm. information (OUT)
)
```

There are several variants available, here the main ones:
- `Waitall`
- `Testall`
- `Waitany`
- `Testany`

>[!example] Example
>**Problem** → ring: each rank sends something to left/right rank, and receives something from them
>
>![[Pasted image 20251008162902.png]]
>
>It could be helpful using a non-blocking send/receive because there could be deadlock problems (if one core is sending to the next one and the next one is sending to the previous one at the same time with sufficiently large messages, there will be a deadlock)
>
>>[!done]- Solution
>>```c
>>#include "mpi.h"
>>#include <stdio.h>
>>
>>int main(void) {
>>	int numtasks, rank, next, prev, buf[2];
>>	MPI_Request reqs[4]; // required variable for non-blocking calls
>>	MPI_Status stats[4]; // required variable for Waitall routine
>>	MPI_Init(NULL, NULL);
>>	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
>>	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // determine left and right neighbors
>>	prev = (rank-1+numtask) % numtasks;
>>	next = (rank+1) % numtasks;
>>// post non-blocking receives and sends for neighbors
>>	MPI_Irecv(&buf[0], 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &reqs[0]);
>>	MPI_Irecv(&buf[1], 1, MPI_INT, next, 0, MPI_COMM_WORLD, &reqs[1]);
>>	MPI_Isend(&rank, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &reqs[2]);
>>	MPI_Isend(&rank, 1, MPI_INT, next, 0, MPI_COMM_WORLD, &reqs[3]);
>>	// do some work while sends/receives progress idn background
>>	// wait for all non-blocking operations to complete
>>	MPI_Waitall(4, reqs, stats);
>>	// continue - do more work
>>	MPI_Finalize();
>>}
>>```

### Summary

| A sending process                               | Function    |
| ----------------------------------------------- | ----------- |
| must block untile the message is delivered      | `MPI_Ssend` |
| should wait only until the message is buffered  | `MPI_Bsend` |
| should return immediately without ever blocking | `MPI_Isend` |
