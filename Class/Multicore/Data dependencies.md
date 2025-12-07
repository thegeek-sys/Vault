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

In this case, reasoning on dependencies the same way as the previous method can be hard