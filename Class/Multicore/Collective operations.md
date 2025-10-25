---
Class: "[[Multicore]]"
Related:
---
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
>    int my_rank, comm_sz, n, local_n;
>    double a, b, local_a, local_b;
>    double local_int, total_int;
>	
>    MPI_Init(NULL, NULL);
>    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
>    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
>	
>    Get_input(my_rank, comm_sz, &a, &b, &n);
>	
>    h = (b - a) / n;         /* h is the same for all processes */
>    local_n = n / comm_sz;   /* So is the number of trapezoids */
>	
>    /* Length of each process' interval of 
>     * integration = local_n * h. So my interval 
>     * starts at: */
>    local_a = a + my_rank * local_n * h;
>    local_b = local_a + local_n * h;
>    local_int = Trap(local_a, local_b, local_n, h);
>	
>    /* Add up the integrals calculated by each process */
>    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
>	
>    /* Print the result */
>    if (my_rank == 0) {
>        printf("With n = %d trapezoids, our estimate\n", n);
>        printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
>    }
>	
>    /* Shut down MPI */
>    MPI_Finalize();
>	
>    return 0;
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
