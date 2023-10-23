---
Created: 2023-10-23
Programming language: "[[Python]]"
Related:
  - "[[list]]"
  - "[[list methods]]"
Completed:
---
---
## Introduction
La **List Comprehension** ci offre una sintassi più corta e veloce per quando di vuole creare una nuova lista basata sui valori di una lista già esistente (rispetto al consueto `for`)

```python
values = [1, 2, 3]
processed = [v**2 for v in values]
print(processed) # -> [1, 4, 9]
```

Ma possono essere usate anche per eseguire operazioni con più valori da inserire insieme e possono essere concatenate con degli `if`

```python
out = []
for val1 in seq1:
    for val2 in seq2:
        if <condizione_usa_val1_val2>:
            out.append(val1, val2)

out = [ (val1, val2) for val1 in seq1 for val2 in seq2 if <condizione_usa_val1_val2>]

# Calcolare tutte le coppie fra (1, 2, 3) e (1, 2, 3) e metterle in una lista SOLO se la
# somma degli elementi nelle coppia è multiplo di 3
T = (1, 2, 3)
out = [(el, el2) for el in T for el2 in T if (el+el2) % 3 == 0]
print(out) # -> [(1, 2), (2, 1), (3, 3)]
```