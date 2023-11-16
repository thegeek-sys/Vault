---
Created: 2023-11-16
Programming language: "[[Python]]"
Related:
  - "[[Programmazione funzionale]]"
Completed:
---
---
## Introduction
La funzione `filter()` ritorna un iteratore su tutta la sequenza di elementi che rispetta una determinata condizione (viene duplicata in modo simile alla list comprehension). Un **predicato** è una funzione che ritorna il vero valore di una determinata condizione; in `filter()`, il predicato può e deve prendere un solo valore.

```python
def is_even(x):
	return (x % 2) == 0
list(filter(is_even, range(10))) # -> [0, 2, 4, 6, 8]
```