---
Created: 2023-11-29
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Una funzione si dice **ricorsiva** quando nella sua definizione la funzione chiama sé stessa
Innanzitutto è importante fare distinzione tra e *Queue* e *Stack*

```start-multi-column
ID: ID_v5fv
Number of Columns: 2
Column Size: [43%, 56%]
```

### Queue
![[queue.png]]

Nella queue vige la regola **first in, last out**

--- column-end ---

### Stack
![[stack.png]]

Nello stack vige la regola **first in, first out**

--- end-multi-column
#### Stack di un programma
![[stack program.png]]

## Sequenza Fibonacci - ricorsivo
```python
def fibonacci(n):
	if n < 2:
		return 1
	else:
		return fibonacci(n-1) + fibonacci(n-2)
```

![[fibonacci.png]]

