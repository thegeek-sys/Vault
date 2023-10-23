---
Created: 2023-10-23
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Index

1. [[#Introduction|Introduction]]
2. [[#Caratteristiche dei set|Caratteristiche dei set]]
3. [[#Operazioni sui set|Operazioni sui set]]

---
## Introduction
I tipi `set` in Python sono quelli che in matematica sono definiti **insiemi** infatti:
- non hanno ordine
- non hanno duplicati
- sono mutabili, ma possono contenere solo elementi immutabili (non posso fare `set` di `list`)
Per definirle si possono usare parentesi graffe `{}` oppure `set()`. Come tali i `set` sono dei tipi iterabili

> [!Warning]
> NOTA: Per creare set vuoti NON si può usare `{}` in quanto questa struttura dati crea un dizionario

```python
my_set = {2, 1, 3 ,4}
print(type(my_set), my_set) # -> <class 'set'> {1, 2, 3, 4}
```

Posso inoltre creare un `set` che prenda in input una sequenza

```python
my_set = set([1, 2, 3, 4, 4, 1])
print(type(my_set), my_set) # -> <class 'set'> {1, 2, 3, 4}
```

---
## Caratteristiche dei set
- Sono veloci a **testare appartenenza** di un elemento all'insieme $\approx \mathcal{O}(1)$ (per testarlo posso utilizzare la funzione `%timeit <istruzione_python>`) attraverso l’operatore `in`
- **Eliminare elementi duplicati**
- Sono utili a fare **operazioni di insiemistica** (unioni, intersezioni etc.)

---
## Operazioni sui set
```python
s = {1, 4, 6, 7}
t = {1, 2, 3, 4, 7, 9}
```
### Unione $\mathcal{A} \cup \mathcal{B}$
```python
s | t # -> {1, 2, 3, 4, 6, 7, 9}
```

### Intersezione $\mathcal{A} \cap \mathcal{B}$
```python
s & t # -> {1, 4, 7}
s.intersection(t)
```

### Differenza $\mathcal{A}~ \backslash  ~\mathcal{B}$
```python
s - t # -> {6}
```

### Differenza simmetrica (XOR) $\{\mathcal{A}~ \backslash  ~\mathcal{B}\} \cup  \{\mathcal{B}~ \backslash  ~\mathcal{A}\}$
Elementi di $\mathcal{A}$ non in $\mathcal{B}$ uniti agli elementi di $\mathcal{B}$ non in $\mathcal{A}$
```python
s ^ t # -> {2, 3, 6, 9}
s.symmetric_difference(t)
```

---
## Metodi
![[set methods]]
