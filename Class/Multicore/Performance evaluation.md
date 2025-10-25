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

>[!question]- Is every rank going to start at the same time?
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
>
>>[!question] What’s the impact of noise in practice?
>>![[Pasted image 20251026005634.png|300]]
>>
>>Intuitively, the more ranks you have, the more likely it is that at least one of them is affected by noise

---
## Example: run-times of serial and parallel matrix-vector multiplication


![[Pasted image 20251026005908.png|470]]

- the runtime increases with the problem size
- the runtime decreases with the number of processes

>[!question] What expectations do we have?
>Ideally, when running with $p$ processes, the program should be $p$ times faster than when running with $1$ process
>
>Let’s define with $T_{\text{serial}}(n)$ the time of our sequential application on a problem of size $n$ (e.g. $n$ is the dimension of the matrix)
>Let’s define with $T_{\text{parallel}}(n-p)$ the time of our parallel application on a problem of size $n$, when running with $p$ processes
>Let’s define with $S(n,p)$ the **speedup** of our parallel application
>
>$$S(n,p)=\frac{T_{\text{serial}}(n)}{T_{\text{parallel}}(n,p)}$$
>
>Thus, ideally, we would like to have $S(n,p)=p$. In this case, we say our program has a *linear speedup*
>
>![[Pasted image 20251026010507.png]]
>
>In general, we expect the speedup to get better when increasing the problem size $n$

>[!info] $T_{\text{serial}}(n)\neq T_{\text{parallel}}(n,1)$
>- $T_{\text{serial}}(n)$ is the time of our sequential application on a problem of size $n$
>- $T_{\text{parallel}}(n,1)$ is the time of our parallel application on a problem of size $n$, when running with one process
>
>These two implementations might be different; in general $T_{\text{parallel}}(n,1)\geq T_{\text{serial}}(n)$
>
>We define **scalability** in this way:
>$$S(n,p)=\frac{T_{\text{parallel})}(n,1)}{T_{\text{parallel}}(n,p)}$$

### Speedups of parallel matrix-vector multiplication
$$S(n,p)=\frac{T_{\text{serial}}(n)}{T_{\text{parallel}}(n,p)}$$

![[Pasted image 20251026010641.png|410]]
### Efficiencies of parallel matrix-vector multiplication
Here’s the definition of **efficiency**:
$$
E(n,p)=\frac{S(n,p)}{p}=\frac{T_{\text{serial}}(n)}{p\cdot T_{\text{parallel}}(n,p)}
$$

Ideally, we would like to have $E(n,p)=1$, but in practice it is $\leq 1$, and it gets worse with smaller problem sizes

![[Pasted image 20251026011816.png|500]]

![[Pasted image 20251026011839.png|410]]

---
## Strong vs. weak scaling
**Strong scaling**
Fix the problem size, and increase the number of processes. If we can keep a high efficiency, our program is *strong scalable*.

**Weak scaling**
Increase the problem size at the same rate at which you increase the number of processes (e.g. every time you  increase the number of processes by 2x, increase also the problem size by 2x). If we can keep a high efficiency, our program is *weak scalable*.

>[!example] From speedup data
>![[Pasted image 20251026012229.png|420]]
>
>Weakly scalable

>[!example] From efficiency data
>![[Pasted image 20251026012326.png|420]]
>Not strongly scalable
>
>![[Pasted image 20251026012401.png|420]]
>Weakly scalable

