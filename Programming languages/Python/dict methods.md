---
Created: 2023-10-23
Programming language: "[[Python]]"
Related:
  - "[[dict]]"
Completed:
---
---
#### `d1.update(d2)`
Il metodo `update` mi permette di aggiungere in-place un dizionario ad un altro dizionario (in caso di elementi con stessa chiave questa verrà sostituita)

#### `dict.get(key, val_def)`
Il metodo `get` rende il valore mappato della chiave `key` **se esiste la chiave altrimenti rende** `val_def`
```python
stringa = 'supercalifragilistichespiralidoso'

counts_b = {}
for char in stringa:
    counts_b[char] = counts_b.get(char,0) + 1
counts_b
assert counts_b == counts
```