---
Class: "[[Multicore]]"
Related:
---
---
## Index
- [[#$ verb MPI_Reduce $|MPI_Reduce]]
	- [[#$ verb MPI_Reduce $ operators|MPI_Reduce operators]]
- [[#$ verb MPI_Bcast $|MPI_Bcast]]
- [[#$ verb MPI_Allreduce $|MPI_Allreduce]]
- [[#Relevance of collective algorithms|Relevance of collective algorithms]]
---

>[!warning] Caveats
>*All* the processes in the communicator must call the same collective function. For example, a program that attempts to match a call to `MPI_Reduce` on one process with a call to `MPI_Recv` on another process is erroneous, and, in all likelihood, the program will hand or crash
>
>The arguments passed by each process to an MPI collective communication must be “compatible”. For example if one process passes in $0$ as the `dest_process` and another passes in $1$, then the outcome of a call to `MPI_Reduce` is erroneous, and, once again, the program is likely to hang or crash
>
>The `output_data_p` argument is only used on `dest_process`. However, all of the processes still need to pass in an actual argument corresponding to `output_data_p`, even if it’s just `NULL`
>
>Point-to-point communications are matched on the basis of tags and communicators, collective communications don’t use tags. They are matched solely on the basis of the communicator and the order in which they’re called

## $\verb|MPI_Reduce|$
![[Pasted image 20251025231822.png]]

```c
MPI_Reduce(
    void*        input_data_p,  // in
    void*        output_data_p, // out
    int          count,         // in
    MPI_Datatype datatype,      // in
    MPI_Op       operator,      // in
    int          dest_process,  // in
    MPI_Comm     comm           // in
);
```

>[!info] Use
>```c
>MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
>```

>[!example]-
>```c
>int main(void) {
>   int my_rank, comm_sz, n, local_n;
>   double a, b, local_a, local_b;
>   double local_int, total_int;
>	
>   MPI_Init(NULL, NULL);
>   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
>   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
>	
>   Get_input(my_rank, comm_sz, &a, &b, &n);
>	
>   h = (b - a) / n;         /* h is the same for all processes */
>   local_n = n / comm_sz;   /* So is the number of trapezoids */
>	
>   /* Length of each process' interval of 
>    * integration = local_n * h. So my interval 
>    * starts at: */
>   local_a = a + my_rank * local_n * h;
>   local_b = local_a + local_n * h;
>   local_int = Trap(local_a, local_b, local_n, h);
>	
>   /* Add up the integrals calculated by each process */
>   MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
>	
>   /* Print the result */
>   if (my_rank == 0) {
>       printf("With n = %d trapezoids, our estimate\n", n);
>       printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
>   }
>	
>   /* Shut down MPI */
>   MPI_Finalize();
>	
>   return 0;
>}
>```

### $\verb|MPI_Reduce|$ operators

| Operation value | Meaning                         |
| --------------- | ------------------------------- |
| `MPI_MAX`       | maximum                         |
| `MPI_MIN`       | minimum                         |
| `MPI_SUM`       | sum                             |
| `MPI_PROD`      | product                         |
| `MPI_LAND`      | logical and                     |
| `MPI_BAND`      | bitwise and                     |
| `MPI_LOR`       | logical or                      |
| `MPI_BOR`       | bitwise or                      |
| `MPI_LXOR`      | logical exclusive or            |
| `MPI_BXOR`      | bitwise exclusive or            |
| `MPI_MAXLOC`    | maximum and location of maximum |
| `MPI_MINLOC`    | minimum and location of minimum |

---
## $\verb|MPI_Bcast|$
Data belonging to a single process is sent to all the processes in the communicator

```c
int MPI_Bcast(
	void*        data_p,      // in/out
	int          count,       // in
	MPI_Datatype datatype,    // in
	int          source_proc, // in, who sends
	MPI_Comm     comm         // in
);
```

>[!example]-
>In the following example the process with rank 0 sends the values to the other processes, which will find the values in the same variables use for sending
>
>```c
>void Get_input(
>			   int my_rank, // in
>			   int comm_sz, // in
>			   double* a_p, // out
>			   double* b_p, // out
>			   int*    n_p, // out
>			   ) {
>	if (my_rank == 0) {
>		printf("Enter a, b, and n\n");
>		scanf("%lf %lf %d", a_p, b_p, n_p);
>	}
>	MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
>	MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
>	MPI_Bcast(b_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
>}
>```

---
## $\verb|MPI_Allreduce|$
Conceptually, is an `MPI_Reduce` followed by `MPI_Bcast` (i.g. compute a global sum and distribute the result to all the processes)

![[Pasted image 20251022142202.png|center|300]]

```c
int MPI_Allreduce(
	void*        input_data_p,  // in
	void*        output_data_p, // out
	int          count,         // in
	MPI_Datatype datatype,      // in
	MPI_Op       operator,      // in
	MPI_Comm     comm           // in
);
```

>[!info]
>The argument list id identical to that for `MPI_Reduce`, except that there is no `dest_process` since all the processes should get the result

And it would act like this and would take $2\cdot \log_{2}(p)\cdot T_{\text{send}}$ (time of the send approximated to the time of the recv)
![[Pasted image 20251022142701.png|300]]

>[!question] Is this the best way of doing it?
>Another way of doing it would be this one
>
>![[Pasted image 20251022143012.png|center|400]]
>This is also known as butterfly pattern (sometimes as recursive distance doubling)
>
>$$T=\log_{2}(p)\cdot T_{\text{send}}$$
>
>It is two times faster (other algos might be better depending on the data size)

---
## $\verb|MPI_Scatter|$
`MPI_Scatter` can be used in a function that reads in an entire vector on process 0 but only send the needed components to each of the other processes (i.g. given a vector, rank 0 splits it and sends one piece to each process)

>[!question] What if I want to send a different number of element to each rank?
>`MPI_Scatterv`. This function can be also helpful even if the number of elements that we are sending is not divisible by the number of processes. In fact the remaining elements (not assigned to any process) will be trashed

![[Pasted image 20251029002615.png|center]]

```c
int MPI_Scatter(
	void*        send_buf_p, // in
	int          send_count, // in
	MPI_Datatype send_type,  // in
	void*        recv_buf_p, // out
	int          recv_count, // in
	MPI_Datatype recv_type,  // in
	int          src_proc,   // in
	MPI_Comm     comm        // in
);
```

>[!warning]
>`send_count` is the number of elements to send to each process, not the total number of elements!

>[!example]
>```c
>MPI_Scatter(buff, 3, MPI_INT, dest, 3, MPI_INT, 0, MPI_COMM_WORLD);
>```
>
>![[Pasted image 20251029003404.png]]

### In place

>[!example]
>```c
>if (rank == 0)
>	MPI_Scatter(buff, 3, MPI_INT, MPI_IN_PLACE, 3, MPI_INT, 0, MPI_COMM_WORLD);
>else
>	MPI_Scatter(buff, 3, MPI_INT, dest, 3, MPI_INT, 0, MPI_COMM_WORLD);
>```
>
>![[Pasted image 20251029003618.png]]

---
## $\verb|MPI_Gather|$
Collect all of the components of the vector onto process 0, and then process 0 can process all of the components

![[Pasted image 20251029180322.png|center|350]]

```c
int MPI_Gather (
	void*        send_buf_p, // in
	int          send_count, // in
	MPI_Datatype send_type,  // in
	void*        recv_buf_p, // out
	int          recv_count, // in
	MPI_Datatype recv_type,  // in
	int          dest_proc,  // in
	MPI_Comm     comm        // in
);
```

>[!warning]
>`send_count` represents the number of elements that each process sends, not the total number of elements in the final vector

>[!example]
>![[Pasted image 20251029180757.png|350]]
>
>![[Pasted image 20251029180834.png|350]]

---
## Collectives on matrices
If we are using a statically allocated matrix, they are stored in contiguous memory, and for this reason it is possible to use any collective operation without any problem (just like arrays)

>[!example]
>```c
>int matrix[3][3];
>```
>
>![[Pasted image 20251030165239.png]]
>
>```c
>MPI_Reduce(sendbuf, recvbuf, num_rows*num_cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
>```

### Gather/Broadcast on dynamically allocated matrix
Let’s allocate a dynamic matrix:
```c
int** a;
a = (int**) malloc(sizeof(int*)*num_rows);
for (int i=0; i<num_rows; i++) {
	a[i] = (int*) malloc(sizeof(int)*num_cols)
}
```

This is how the matrix would look like in memory:
![[Pasted image 20251030165515.png|400]]

>[!error] Wrong
>```c
>// after the first num_rows elements we don't know what there is
>MPI_Reduce(a, recvbuf, num_rows*num_cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
>
>// after the first num_cols of the first row we don't know what there is
>MPI_Reduce(a[0], recvbuf, num_rows*num_cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
>```

>[!done] Correct
>```c
>for (int i=0; i<num_rows; i++) {
>	MPI_Reduce(a[i], recvbuf[i], num_cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
>}
>```

Another useful method would be to **linearize the matrix**. In this case we should allocate the memory like this

```c
int* a;
a = (int*) malloc(sizeof(int)*num_rows*num_cols);
// ...
// ...
// a[i][j]
a[i * num_cols + j] = ...
```

Then apply the `Reduce` like if it would be allocated statically
```c
MPI_Reduce(a, recvbuf, num_rows*num_cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)
```

---
## $\verb|MPI_Allgather|$
Conceptually, it is like a gather + broadcast; in practice it might be implemented in a more efficient way

![[Pasted image 20251031092240.png|300]]

```c
MPI_Allgather(
	void*        send_buf_p, // in
	int          send_count, // in
	MPI_Datatype send_type,  // in
	void*        recv_buf_p, // out
	int          recv_count, // in
	MPI_Datatype recv_type,  // in
	MPI_Comm     comm        // in
);
```

>[!warning] Attention
>`send_count` and `recv_count` are the number of elements sent by each process

---
## Relevance of collective algorithms
Collective algorithms are widely used in large-scale parallel applications from many domains as they account for a large fraction of the total runtime and they are highly relevant for distributed training of deep-learning models

That’s the reason why all the big players are designing their own collective communication library. For example:
- NCCL → NVIDIA
- RCCL → AMD
- OnceCCL → Intel
- MSCCL → Microsoft

>[!question] Given a collective (eg. `MPI_Reduce`), how to select the best algorithm?
>- automatically through heuristic
>- manually
>- MPI implementations such as Open MPI do not make assumption on the underlying hardware 

It is an active research area, both from algorithmic and implementations standpoints

