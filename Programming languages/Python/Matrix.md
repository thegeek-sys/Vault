---
Created: 2023-11-20
Programming language: "[[Python]]"
Related:
  - "[[Matrix]]"
  - "[[Shallow and Deep Copy]]"
  - "[[Images]]"
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
# col col
# [0, 1] row
# [1, 2] row
# [2, 3] row
```

---
## Funzione per creare una matrice

```python
def create_matrix(r,c,value=0):
	# define matrix
	matrix = []
	# for each row
	for each_r in range(r):
		# define row
		row = []
		# for each row
		for each_c in range(c):
			# append the col to the row
			row.append(value)
		# append the row to the matrix
		matrix.append(row)
	return matrix

def create_matrix_lc(r,c,value=0):
	return [ [value] * c for each_r in range(r) ]

def create_matrix_map(r,c,value=0):
	return list(map(lambda each_c))

create_matrix(20,20)
```

  Potremmo anche pensare di crearla con `[ [0] * c] * r`, ma questo non funzionerà poiché se dopo provo a scrivere in un singolo pixel mi scriverà nell’intera colonna corrispondente

---
## Trasporre matrice
```python
matt = [ [1, 2, 3], [ 4, 5, 6], [7, 8, 9] ]

def transpose(im):
    H,W = shape(im)
    # blocco la riga C-estima prendi i valori dalle righe
    # li rimetto come righe nella nuova matrice
    return [ [ im[r][c] for r in range(H)] for c in range(W)  ]
```