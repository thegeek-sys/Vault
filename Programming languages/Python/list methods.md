---
Created: 2023-10-12
Programming language: "[[Python]]"
Related: []
Completed:
---
---
## Methods
#### `list.append()`
Per aggiungere elementi in coda alle liste viene utilizzata il metodo `append()` che modifica la lista **“in-place”** (non la devo riassegnare)

```python
lista = [1,2,3,5]
lista.append(4)
lista += [4] # è òa stessa cosa
print(lista) # -> [1,2,3,5,4]

lista = lista.append(4)
print(lista) # -> None
```

#### `list.extend()`
Il metodo `estend()` serve per aggiungere una lista in coda ad una lista (se provassi a fare la stessa cosa con `append()` mi ritroverà con delle liste concatenate)
```python
t = ['a', 'b', 'c']
s = ['d', 'e']

t.extend(s)
print(t) # -> ['a', 'b', 'c', 'd', 'e']
```

#### `list.remove()`
Toglie il primo elemento che corrisponde all’oggetto dato

#### `list.pop()`
La funzione pop mi permette di 
```python
t = ['a','b','c','d','e']
removed = t.pop(1)
print(removed) # ->
```