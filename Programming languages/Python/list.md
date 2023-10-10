---
Created: 2023-10-10
Programming language: "[[Python]]"
Related:
  - "[[Expression statements]]"
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

Nelle `list`, a differenza della stringhe, `is` e `==` hanno una diversa funzione, infatti facendo una copia di una lista, questa sarà assegnata a una diversa locazione di memoria, nonostante abbia elementi identici (due stringhe uguali sono assegnate alla stessa locazione di memoria)

 ```python
lista_a = [1,2,3,4,5]
lista_b = [6,7,8,9,0]

lista_a+lista_b
```

Per testare l’esistenza di un elemento in una lista utilizziamo [[Comparatori e operatori di appartenenza|in]]
