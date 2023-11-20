---
Created: 2023-11-20
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Una matrice altro non è che un array bidimensionale organizzato per righe e colonne. In Python vengono interpretare come “lista di liste”

![[matrix.jpg]]

```python
matrix_3x2 = [[0,1],[1,2],[2,3]]
r = len(matrix_3x2) # righe
c = len(matrix_3x2[0]) # colonne

def print_matrix(m):
	for r in m:
		print(r)
print_matrix(matrix_3x2)
```