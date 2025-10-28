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

---
## Amdahl’s law

>[!info] Intuition
>Each program has some part of it which cannot be parallelized (serial fraction, $1-\alpha$) like reading/writing a file from disk, sending/receiving data over the network, serialization due to lock/unlock, etc.

![[Pasted image 20251028171035.png|400]]
In this image $p$ is the part that can be parallelized while $s$ is the part that cannot be

Amdahl’s law says that the speedup is limited by the **serial fraction** ($s$) 

$$
T_{\text{parallel}}(p)=(1-\alpha)T_{\text{serial}}+\frac{\alpha T_{\text{seiral}}}{p}
$$

A fraction $0\leq \alpha\leq 1$ can be parallelized while the remaining $1-a$ has to be done sequentially

>[!example]
>- if $\alpha= 0$, the code can’t be parallelized and $T_{\text{parallel}}(p)=T_{\text{serial}}$
>- if $\alpha=1$, the entire code can be parallelized and $T_{\text{parallel}}(p)=\frac{T_{\text{serial}}}{p}$ (ideal speedup)

Now let’s use $T_{\text{parallel}}(p)$ to express the speedup formula:
$$
S(p)=\frac{T_{\text{serial}}}{(1-\alpha)T_{\text{serial}}+\frac{\alpha T_{\text{serial}}}{p}}
$$

The following is the upper asymptotic limit of the speedup to which we can aim:
$$
\lim_{ p \to \infty } S(p)=\frac{1}{1-\alpha}
$$

>[!example]
>- if $60\%$ of the application can be parallelized, $\alpha = 0.6$, which means we can expect a speedup of at most $2.5$
>- if $80\%$ of the application can be parallelized, $\alpha = 0.8$, which means we can expect a speedup of at most $5$
>
>To be able to scale up to $100000$ processes, we need to have $\alpha \geq 0.99999$
>
>Let’s plot the Amdahl’s law:
>![[Pasted image 20251028172251.png]]

---
## Gustafson’s law
If we consider weak scaling (instead of the strong scaling considered in the Amdahl’s law), the parallel fraction increases with the problem size (i.e., the serial time remains constant, but the parallel time increases).

It is also known as **scaled speedup**.
$$
S(n,p)=(1-\alpha)+\alpha p
$$

### Amdahl’s law vs. Gustafson’s law
![[Pasted image 20251028172451.png]]

### Amdahl’s law limitations
The serial fraction could get bigger when increasing the number of processor (i.e. the runtime might increase when increasing the number of processors)

![[Pasted image 20251028172559.png|300]]

---
## Example: sum between vectors
$$
\begin{align}
x+y&=(x_{0},x_{1},\dots,x_{n-1})+(y_{0},y_{1},\dots,y_{n-1}) \\
&=(x_{0}+y_{0},x_{1}+y_{1},\dots,x_{n-1}+y_{n-1}) \\
&=(z_{0},z_{1},\dots,z_{n-1}) \\
&=\pmb{z}
\end{align}
$$
### Serial implementation

```c
void Vector_sum(double x[], double y[], double z[], int n) {
	int i;
	for(i=0; i<n; i++) {
		z[i] = x[i] + y[i];
	}
}
```

### Parallel implementation

```c
void Parallel_vector_sum(
		double local_x[], // in
		double local_y[], // in
		double local_z[], // out
		int    local_n    // in
) {
	int local_i;
	for (local_i=0; local_i<local_n; local_i++)
		local_z[local_i] = local_x[local_i] + local_y[local_i];
}
```

This function could also be optimized by using [[Collective operations#$ verb MPI_Scatter $|MPI_Scatter]]
