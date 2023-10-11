---
Created: 2023-10-10
Programming language: "[[Python]]"
Related:
  - "[[Comparatori e operatori di appartenenza]]"
Completed:
---
---
## Introduction
Le `list`, come le tuple, permettono di elencare degli elementi ma restano mutabili (possiamo modificare gli item)

```python
numbers = [0,10,20,30,4]
numebrs[4] = 40
print(numbers) # -> [0,10,20,30,40]
```

Come le tuple e le stringhe supportano lo slicing però attenzione a come lo si usa. L’operatore `*` viene spesso utilizzato per creare delle liste vuote che verranno successivamente popolate

```python
numbers = [0,10,20,30,4]
numbers[1:2] = [1,2,3]
print(numbers) # -> [0, 1, 2, 3, 10, 20, 30, 4]

numbers[1:4] = [1,2,3]
print(numbers) # -> [0, 1, 2, 3, 4]

[None]*5 # -> [None, None, None, None, None]
```

Nelle `list`, a differenza della stringhe, `is` e `==` hanno una diversa funzione, infatti facendo una copia di una lista, questa sarà assegnata a una diversa locazione di memoria, nonostante abbia elementi identici (due stringhe uguali sono assegnate alla stessa locazione di memoria). Le liste sono inoltre valutate `False` solo quando la lista è vuota

 ```python
lista_a = [1,2,3,4,5]
lista_b = [6,7,8,9,0]

lista_a+lista_b
bool([0]) # -> True
```

Per testare l’esistenza di un elemento in una lista utilizziamo [[Comparatori e operatori di appartenenza#`in`|in]]

```python
'python' in ['c', 'js', 'assembly', 'Python'].lower() # -> True
```

---
## `append()`
Per aggiungere elementi in coda alle liste viene utilizzata il metodo `append()` che modifica la lista **“in-place”** (non la devo riassegnare)

```python
lista = [1,2,3,5]
lista.append(4)
print(lista) # -> [1,2,3,5,4]

lista = lista.append(4)
print(lista) # -> None
```

---
## `extend()`
Il metodo `estend()` serve per aggiungere più valori in coda ad una lista (`append` mi permette di aggiungerne solo uno)
```python
t = ['a', 'b', 'c']
s = ['d', 'e']
t.extend(s)
print(t) # -> ['a', 'b', 'c', 'd', 'e']
```
