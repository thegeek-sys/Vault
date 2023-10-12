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
print(lista) # -> [1,2,3,5,4]

lista = lista.append(4)
print(lista) # -> None
```

#### `list.extend()`
Il metodo `estend()` serve per aggiungere più valori in coda ad una lista (`append` mi permette di aggiungerne solo uno)
```python
t = ['a', 'b', 'c']
s = ['d', 'e']
t.extend(s)
print(t) # -> ['a', 'b', 'c', 'd', 'e']
```

