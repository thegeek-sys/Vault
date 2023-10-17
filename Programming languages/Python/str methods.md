---
Created: 2023-10-11
Programming language: "[[Python]]"
Related:
  - "[[str]]"
Completed:
---
---
## Methods
#### `str.strip()`
Il metodo `strip()` rimuove eventuali spazi all’inizio e alla fine di una stringa
```python
s = '    ciao  '
print(s.strip()) # -> 'ciao'
```

#### `str.split(str)`
Il metodo `split()` divide una stringa in corrispondenza del separatore specificato e restituisce una `list` di sottostringhe. Se non metto alcun argomento sceglierà di default `' '`
```python
s = 'ciao, come, stai'
print(s.split(', ')) # -> ['ciao', 'come', 'stai']
```

#### `str.replace(str, str)`
Il metodo `replace()` sostituisce una frase specificata (ogni volta che si presenta nella stringa) con un'altra frase specificata

#### `str.zfill(int)`
Il metodo `zfill()` aggiunge alla stringa tanti zeri sulla sinistra fino a raggiungere il numero indicato in `int`