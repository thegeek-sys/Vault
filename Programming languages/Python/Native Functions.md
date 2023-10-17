---
Created: 2023-10-11
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Functions
#### `ord(str)`
La funzione `ord()` ritorna il codice Unicode di un carattere dato come argomento

#### `chr(int)`
La funzione `chr()` ritorna il carattere corrispondente all’Unicode dato come argomento

#### `isinstance(obj, class)`
La funzione `isinstance()` ritorna `True` se l’object specificato è del tipo specificato, altrimenti `False`.

#### `sum(list)`
La funzione `sum()` calcola la somma tra tutti gli oggetti di una lista

#### `max(list)`
La funzione `max()` mi restituisce l’oggetto con valore maggiore in una lista

#### `min(list)`
La funzione `min()` mi restituisce l’oggetto con valore minore in una lista

#### `enumerate(list)`
Se vogliamo accedere all’elemento della lista e al suo corrispettivo indice ed enumerate mi permette di spacchettare una tupla in cui sono presenti indice ed oggetto di una lista
```python
numb = [-1,-5,2,592,700]

for i, elem in enumerate(numb):
	print(i,')',elem)

# ->
# 0 ) -1
# 1 ) -5
# 2 ) 2
# 3 ) 592
# 4 ) 700

numb = [(1,2),]*10
for i, (dx, sx) in enumerate(numb):
# sto mettendo le parentesi per far capire a python che sto analizzando una tupla
	print(i,')',dx,sx) # -> 0 ) 1 2
```

