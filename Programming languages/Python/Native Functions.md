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

#### `zip(src, dst)`
La funzione `zip` è un generatore che mi rende, date due sequenze, le **coppie allineate**. Di fatti prende due sequenze (della stessa lunghezza) e mi rendere una tupla con un carattere proveniente dal primo tipo iterabile e un carattere proveniente dal secondo tipo iterabile.
Se sono di lunghezza diversa zip si interrompe sulla lista più corta

```python
languages = ['Java', 'Python', 'JavaScript']
versions = [14, 3, 6]

result = zip(languages, versions)
print(list(result)) # -> [('Java', 14), ('Python', 3), ('JavaScript', 6)]
```