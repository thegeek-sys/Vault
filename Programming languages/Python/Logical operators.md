---
Created: 2023-10-10
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
La precedenza di esecuzione degli operatori booleani è:
1. `not`
2. `and`
3. `or`

---
## `and`
Ritorna `True` se entrambe le espressioni sono vere

---
## `or`
Ritorna `True` se una delle due espressioni è vera

---
## `not`
Inverte il risultato, ritorna `False` se l’espressione era `True` e viceversa

L’ordine valutazione operatori è:
1. `not`
2. `and`
3. `or`

```python
True and not False or True and False or not True # -> True
```
