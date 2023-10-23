---
Created: 2023-10-23
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
I tipi `set` in Python sono quelli che in matematica sono definiti **insiemi** infatti:
- non hanno ordine
- non hanno duplicati
- sono mutabili, ma possono contenere solo elementi immutabili (non posso fare `set` di `list`)
Per definirle si possono usare parentesi graffe `{}` oppure `set()`

> [!Warning]
> NOTA: Per creare set vuoti NON si può usare `{}` in quanto questa struttura dati crea un dizionario

```python
my_set = {2, 1, 3 ,4}
print(type(my_set), my_set) # -> <class 'set'> {1, 2, 3, 4}
```

Posso inoltre creare un `set` che prenda in input una sequenza

```python
my_set = set([1, 2, 3, 4])
print(type(my_set), my_set) # -> <class 'set'> {1, 2, 3, 4}
```
## Caratteristiche dei set
- Sono veloci a **testare appartenenza** di un elemento all'insieme $\approx \mathcal{O}(1)$ (per testarlo posso utilizzare la funzione `%timeit <istruzione_python>`) attraverso l’operatore `in`
- **Eliminare elementi duplicati**
- Sono utili a fare **operazioni di insiemistica** (unioni, intersezioni etc.)
