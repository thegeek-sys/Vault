---
Created: 2023-10-12
Programming language: "[[Python]]"
Related:
  - "[[list methods]]"
  - "[[list]]"
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
Il metodo `estend()` serve per aggiungere una lista in coda ad una lista (se provassi a fare la stessa cosa con `append()` mi ritroverò con delle liste concatenate)
```python
t = ['a', 'b', 'c']
s = ['d', 'e']

t.extend(s)
print(t) # -> ['a', 'b', 'c', 'd', 'e']
```

#### `list.remove()`
Il metodo `remove()`serve il primo elemento che corrisponde all’oggetto dato
```python
t = ['a', 'b', 'c', 'd', 'e', 'b']
t.remove('b')
print(t) # -> ['a', 'c', 'd', 'e', 'b']
```

#### `list.pop()`
Il metodo `pop()` mi permette di eliminare l’elemento corrispondente all’indice specificato e eventualmente di poterlo assegnare ad una variabile
```python
t = ['a', 'b', 'c', 'd', 'e']
removed = t.pop(1)
print(t,removed) # -> ['a', 'c', 'd', 'e] b
```

#### `del list[int]`
La funzione `del` eliminerà l’indice `int` della lista data
```python
t = ['a', 'b', 'c', 'd', 'e']
del t[0]
print(t) # -> ['b', 'c', 'd', 'e]
```

#### `list.clear()`
Il metodo `clear()` ci permette di rimuovere tutti gli elementi di una lista in-place

#### `list.sort()`
Il metodo `sort()` ci permette di ordinare in modo **stabile** (l’ordine parziale di due elementi uguali è mantenuto) tutti gli elementi di una lista in-place in modo crescente.
> [!WARNING]
> Possono essere ordinati solo stessi tipi


#### `sorted(list)`
La funzione `sorted()` ci permette di creare una copia ordinata di tutti gli elementi di una lista

#### `lista.reverse()`
Il metodo `reverse()` ci permette di invertire l’ordine di tutti gli elementi di una lista in-place

#### `reversed(list)`
La funzione `reversed()` ci permette di creare una copia ordinata al contrario di tutti gli elementi di una lista