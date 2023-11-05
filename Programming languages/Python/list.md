---
Created: 2023-10-10
Programming language: "[[Python]]"
Related:
  - "[[list methods]]"
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

Come le tuple e le stringhe supportano lo slicing però attenzione a come lo si usa, può essere infatti anche usato come metodo per copiare una lista o sostituirla del tutto (l’operatore `=` creerà solamente un alias). L’operatore `*` viene spesso utilizzato per creare delle liste vuote che verranno successivamente popolate.

```python
numbers = [0,10,20,30,4]
numbers[1:2] = [1,2,3]
print(numbers) # -> [0, 1, 2, 3, 10, 20, 30, 4]
new_numbers = [8,9,10,11,12]
numbers[:] = new_number # sostiuisco numbers nella sua interezza 
						# con il contenuto di new_number in modo
						# tale da mantenere invariato l'id di
						# di memoria di numbers

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
## Methods
![[list methods]]


---
## Complessità delle operazioni su lista

| Operation        | Examples           | Complexity class |                      |
| ---------------- | ------------------ | ---------------- | -------------------- |
|                  |                    | Average case     | Amortised Worst case |
| Append           | `l.append(item)`   | $O(1)$           | $O(1)$               |
| Clear            | `l.clear()`        | $O(1)$           | $O(1)$               |
| Containment      | `item in/not in l` | $O(N)$           | $O(N)$               |
| Copy             | `l.copy()`         | $O(N)$           | $O(N)$               |
| Delete           | `del l[i]`         | $O(N)$           | $O(N)$               |
| Extend           | `l.extend(…)`      | $O(N)$           | $O(N)$               |
| Equality         | `l1==l2, l1!=l2`   | $O(N)$           | $O(N)$               |
| Index            | `l[i]`             | $O(1)$           | $O(1)$               |
| Iteration        | `for item in l:`   | $O(N)$           | $O(N)$               |
| Length           | `len(l)`           | $O(1)$           | $O(1)$               |
| Multiply         | `k*l`              | $O(k*N)$         | $O(k*N)$             |
| Min, Max         | `min(l), max(l)`   | $O(N)$           | $O(N)$               |
| Pop from end     | `l.pop(-1)`        | $O(1)$           | $O(1)$               |
| Pop intermediate | `l.pop(item)`      | $O(N)$           | $O(N)$               |
| Remove           | `l.remove(…)`      | $O(N)$           | $O(N)$               |
| Reverse          | `l.reverse()`      | $O(N)$           | $O(N)$               |
| Slice            | `l[x:y]`           | $O(y-x)$         | $O(y-x)$             |
| Sort             | `l.sort()`         | $O(N*log(N))$    | $O(N*log(N))$        |
| Store            | `l[i]=item`        | $O(1)$           | $O(1)$               |
