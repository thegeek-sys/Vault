---
Class: "[[Multicore]]"
Related:
---
---
## Introduction
A data dependence occurs when a program instruction depends from the result of another previous instruction. In a parallel context, this it critical, because it breaks the logical chain: if two operations have to occur in a specific sequence, they can’t be executed simultaneously by different threads without precautions.

Let’s analyze this example:
```c
fibo[0] = fibo[1] = 1;

#pragma omp parallel for num_threads(2)
for (i = 2; i < n; i++)
	fibo[ i ] = fibo[ i – 1 ] + fibo[ i – 2 ];
```

Usually we get as output `1 1 2 3 5 8 13 21 34 55` which is correct, but sometimes we can get something like `1 1 2 3 5 8 0 0 0 0`

>[!question] What happened?
>1. OpenMP compilers don’t check for dependencies among iterations in a loop that’s being parallelized with a `parallel for` directive
>2. a loop in which the results of one or more iterations depend on other iterations cannot, in general, be correctly parallelized by OpenMP; we say that we have a **loop-carried dependence**

Assuming we have a loop of the form:
```c
for (i=...) {
	S1 : operate on a memory location x
	...
	S2 : operate on a memory location x
}
```

There are four different ways that `S1` and `S2` are connected, based on whether they are reading or writing to `x`

---
## Dependence types
**Flow dependence** → *RAW* (read after write)
```c
// if S2 is executed before S1, S2 will use previous value of x
x = 10;     // S1
y =  2*x+5; // S2
```

**Anti-flow dependence** → *WAR* (write after read)
```c
// if S2 is executed before S1, S1 will use incremented value of x
y = x+3; // S1
x++;     // S2
```

**Output dependence** → *WAW* (write after write)
```c
// if S2 is executed before S1, the final value of x could be overwritten from an incorrect instruction
x = 10;  // S1
x = x+c; // S2
```

**Input dependence** → *RAR* (it’s not an actual dependence, read after read)
```c
y = x+c;   // S1
z = 2*x+1; // S2
```

---
## Flow dependence remval (RAW)
There are 6 techniques to remove this kind of data dependence:
- reduction/induction variable fix
- loop skewing
- partial parallelization
- refactoring
- fissioning
- algorithm change

### Reduction, induction variables
Example:
```c
double v = start;
double sum = 0;
for (int i=0; i<N; i++) {
	sum = sum + f(v); // S1
	v = v + step;     // S2
}
```

- RAW (S1) → caused by reduction of variable `sum`
- RAW (S2) → caused by *induction variable* `v` (induction variable is a variable that gets increased/decreased by a constant amount at each iteration)
- RAW (S2→S1) → caused by induction variable `v`

>[!warning]
>RAW are between the $i$ and the $i+1$ iterations (e.g. `sum` is read in the $(i+1)$-th iteration after being written in the $i$-th iteration)

Remove RAW (S2) and RAW (S2→S1)
```c
double v;
double sum = 0;
for(int i = 0; i < N ; i++) {
	// v does not increment itself
	v = start + i*step;
	sum = sum + f(v);
}

// i=0 -> v = start
// i=1 -> v = start+step
// i=2 -> v = (start+step)+step
```

Remove RAW (S1)
```c
double v;
double sum =0;
#pragma omp parallel for reduction(+ : sum) private(v)
for(int i = 0; i < N ; i++) {
	v = start + i*step;
	sum = sum + f(v);
}
```

### Loop skewing
Another technique involves the rearrangement of the loop body statements. Let’s analyze the following example:
```c
for (int i=1; i<N; i++) {
	y[i] = f(x[i-1]);   // S1
	x[i] = x[i] + c[i]; // S2
}
```

RAW (S2→S1) on `x`

>[!done] Solution
>Make sure the statements that consume the calculated values cause the dependence, use values generated during the same iteration. So not using `i-1`

```c
y[1] = f(x[0]);
for (int i=1; i<N; i++) {
	x[i] = x[i] + c[i];
	y[i+1] = f(x[i]);
}
x[N-1] = x[N-1] + c[N-1];
```

>[!question] How to do loop skewing?
>>[!tip] Unroll the loop and see the repetition pattern
>
>```c
>y[1] = f(x[0]);
>x[1] = x[1]+c[1];
>y[2] = f(x[1]);
>x[2] = x[2]+c[2];
>...
>y[N-2] = f(x[N-3]);
>x[N-2] = x[N-2]+ c[N-2];
>y[N-1] = f(x[N-2]);
>x[N-1] = x[N-1]+ c[N-1];
>```
>
>In this case if have to group them as it follows
>```c
>y[1] = f(x[0]);
>x[1] = x[1]+c[1]; // group
>y[2] = f(x[1]);   // group
>x[2] = x[2]+c[2];
>...
>y[N-2] = f(x[N-3]);
>x[N-2] = x[N-2]+ c[N-2]; // group
>y[N-1] = f(x[N-2]);      // group
>x[N-1] = x[N-1]+ c[N-1];
>```
>
>So the first and last iterations have to be outside of the loop, while the others have to be grouped in this way inside the loop

### Partial parallelization
ISDG is made of up of nodes that represent a single execution of the loop body, and edges that represent dependencies.

Let’s analyze this example:
```c
for (int i=1; i<N; i++)
	for (int j=1; j<M; j++)
		data[i][j] = data[i-1][j] + data[i-1][j-1];
```

In this case, reasoning on dependencies the same way as the previous method can be hard, but we can draw as a matrix the dependencies between the iterations:
![[Pasted image 20251208000905.png|500]]

In this drawing the arrows represent the dependencies between iterations, and, as we can see, there are no dependencies between nodes of the same row. For this reason we can parallelize the inner loop (that iterates on the row)

```c
for (int i=1; i<N; i++)
	#pragma omp parallel for
	for (int j=1; j<M; j++)
		data[i][j] = data[i-1][j] + data[i-1][j-1];
```

### Refactoring
Refactoring refers to rewriting the loop(s) so that parallelism can be exposed. The ISDG for the following example:
```c
for (int i=1; i<N; i++)
	for (int j=1; j<N; j++)
		data[i][j] = data[i-1][j] + data[i][j-1] + data[i-1][j-1]
```

![[Pasted image 20251203144752.png|500]]

Tho in this case diagonal sets, called *wave*, can be executed in parallel (no edges/dependencies between nodes in the same diagonal set)
![[Pasted image 20251203144919.png|500]]

The solution so should have an outer loop that iterates on the number of waves and an inner loop that iterates on the wave itself (the inner loop can be parallelized)

```c
// intuition
for(wave=0 wave<NumWaves; wave++) {
	diag=F(wave);
	#pragma omp parallel for
	for(k=0; k<diag; k++) {
		int i = get_i(diag, k);
		int j = get_j(diag, k);
		data[i][j] = data[i-1][j] + data[i][j-1] + data[i-1][j-1];
	}
}
```

>[!info]
>The execution in waves requires a change  of loop variables from the original `i` and `j`

### Fissioning
Fissioning means breaking the loop apart into a sequential and a parallelizable part

```c
s = b[0];
for (int i=1; i<N; i++) {
	a[i] = a[i] + a[i-1]; // S1
	s = s + b[i];
}
```

Becomes
```c
// sequential part
for (int i=1; i<N; i++) {
	a[i] = a[i] + a[i-1]; // S1
}

// parallel part
s = b[0];
#pragma omp parallel for reduction(+:a)
for (int i=1; i<N; i++) {
	s = s + b[i];
}
```

If everything else fails, switching the algorithm may be the answer. For example, the Fibonacci sequence:
```c
for (int i=2; i<N; i++) {
	int x = F[i-2]; // S1
	int y = F[i-1]; // S2
	F[i] = x+y;     // S3
}
```

Can be parallelized via Binet’s formula:
$$
F_n = \frac{\varphi^n - (1 - \varphi)^n}{\sqrt{5}}
$$

---
## Antidependence removal (WAR)

```c
for (int i=0; i<N-1; i++) {
	a[i] = a[i-1] + c;
}
```

In this case a thread might execute iteration $i$ while iteration $i+1$ is already been executed, so $a[i]$ will be assigned to the wrong value. A simple solution can be making a copy of a before starting to modify it:
```c
for (int i=0; i<N-1; i++) {
	a2[i] = a[i+1];
}

#pragma omp parallel for
for (int i=0; i<N-1; i++) {
	a[i] = a2[i]+c
}
```

>[!warning]
>Space and time tradeoffs must be carefully evaluated! In this case it’s doesn’t make any sense, we are just adding more complexity

---
## Output Dependence Removal (WAW)

```c
for (int i=0; i<N; i++) {
	y[i] = a*x[i] + c; // S1
	d = fabs(y[i]);    // S2
}
```

In this case I want to guarantee that at the end of the execution, the computed `d` is the one computed in the last iteration. We can do this by using the `lastprivate` directive

```c
#pragma omp parallel for shared(a,c) lastprivate(d)
for (int i=0; i<N; i++) {
	y[i] = a*x[i] + c; // S1
	d = fabs(y[i]);    // S2
}
```