---
Class: "[[Multicore]]"
Related:
---
---
## Elapsed parallel time

```c
double MPI_Wtime(void);
```

Returns the number of seconds that have elapsed since some time in the past

>[!example]
>```c
>double start, finish;
>// ...
>start = MPI_Wtime():
>// code to be timed
>// ...
>finish = MPI_Wtime();
>printf("Proc %d > Elapsed time = %e seconds\n", my_rank, finish-start);
>```

>[!question] Which rank?
>Each rank might finish at a different time
>>[!done] Solution
>>Report the maximum time across the ranks
>
>>[!example]
>>```c
>>double local_start, local_finish, local_elapsed, elapsed;
>>// ...
>>local_start = MPI_Wtime():
>>// code to be timed
>>// ...
>>local_finish = MPI_Wtime();
>>local_elapsed = local_finish-local_start;
>>MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
>>
>>if(my_rank==0) {
>>	printf("Elapsed time = %e seconds\n", elapsed);
>>}
>>```
>
>But even this solution could be not exact, in fact might happen that a process takes a lot because it’s waiting a receive

>[!question] Is every rank going to start at the same time?
>Not necessarily. If not, the time we report might be longer not because the application was performing poorly, but rather because someone started later than someone else
>
>To ensure that they are going to start at the same time we use **`MPI_Barrier`**
>
>>[!question] But `MPI_Barrier` is itself a collective, what if it is implemented with a tree? Every rank could exit the barrier at a different time
>>Guaranteeing that they start exactly at the same time is a bit more complicated. For the purposes of this course, `MPI_Barrier` provides a reasonable approximation

>[!question] Is one run/measurement enough?
>No, performance data is not deterministic
>
>>[!example]
>>If you run your application 100 times, you will get 100 different runtimes (this is also known as *noise*)
>
>>[!question] Why?
>>- on a given compute node, interference from other applications and/or operating system (context switched, cache pollution, etc.) 
>>- across multiple nodes, interference on the network (is a resource shared among multiple nodes and applications)
>>
>>![[Pasted image 20251022162614.png]]
>
>>[!done] Solution
>>Run the application multiple times and report the entire distribution of timing
>>
>>![[Pasted image 20251022162815.png]]

