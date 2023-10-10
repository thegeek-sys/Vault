---
Created: 2023-10-05
Programming language: "[[Python]]"
Related:
  - "[[int]]"
  - "[[str]]"
  - "[[float]]"
  - "[[list]]"
Completed:
---
---
## `is`
L'operatore `is` ci permetterà di confrontare gli indirizzi di memoria a cui sono assegnate due variabili. Questo può essere utile in quanto Python per ottimizzazione assegna allo stesso indirizzo di memoria quando due stringhe sono uguali

```python
nome = 'flavio'
altro_nome = 'flavio'
print(nome is altro_nome) # -> True
```
---
## `==`
L'operatore `==` mi permette di confrontare il contenuto di due variabili
```python
print(id(nome), id(altro_nome), id(nome) == id(altro_nome)) # ci permette di vedere l'indirizzo di memoria assegnato alle due variabili
print(nome == altro_nome) # -> True
```
---
## `in`
L’operatore `in` è un operatore di appartenenza che controlla se 
```python
print('gramma' in 'Fondamenti di programmazione') # -> True
```
