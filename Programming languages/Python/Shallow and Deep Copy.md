---
Created: 2023-11-20
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
L’operatore di assegnamento in Python: non copia gli oggetti ma crea dei «bindings» (collegamenti) fra una variabile e l’oggetto

```start-multi-column
ID: ID_wg2f
Number of Columns: 2
Largest Column: standard
```

### Shallow Copy
Crea una nuova variabile `dst`, che referenzia l’oggetto puntato a `src`

--- column-end ---

### Deep copy
Crea una nuova variabile `dst`, che «copia» (ridondante) il contenuto puntato da `src` (viene effettivamente duplicata la variabile e creata una nuova allocazione di memoria)

--- end-multi-column
Negli oggetti immutabili però la deep copy non esiste, il problema si pone nei tipi mutabili.

---
## Shallow vs Deep Copy – Oggetti Mutabili

```start-multi-column
ID: ID_f7em
Number of Columns: 2
Largest Column: standard
```

### Shallow Copy
Crea un nuovo oggetto composto e inserisce riferimenti nel nuovo oggetto puntato a quelli trovati nell’originale.

--- column-end ---

### Deep copy
Crea un nuovo oggetto composto e poi in maniera ricorsiva, inserire copie ridondanti degli oggetti trovati nell’originale.

--- end-multi-column
```python
# SHALLOW
import copy
l_sc1 = list(l)
l_sc2 = l.copy()
l_sc3 = l[:]
l_sc4 = copy.copy(l)

# DEEP
import copy
l_dc1 = copy.deepcopy(l)
```

Con la shallow copy si duplicherò la lista, ma gli elementi (se mutabili) rimarranno solamente dei puntatori all’originale (se modifico la copia modifico anche l’originale, ma se modifico l’originale la copia rimane immutata), mentre nella deep copy duplicherò la lista e tutti gli elementi in essa contenuti in modo ridondante.
