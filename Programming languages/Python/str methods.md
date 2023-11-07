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
Il metodo `strip()` rimuove eventuali spazi all’inizio e alla fine di una stringa. Il metodo `rstrip()` rimuove eventuali withespaces alla fine di una stringa. Il metodo `lstrip()` rimuove eventuali withespaces all’inizio di una stringa
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
```python
s = 'ciao pippo, ciao pluto'
o = s.replace('ciao', 'hello')
print(o) # -> 'hello pippo, hello pluto'
```

#### `str.zfill(int)`
Il metodo `zfill()` aggiunge alla stringa tanti zeri sulla sinistra fino a raggiungere il numero indicato in `int`
```python
s = '5'
o = s.zfill(3)
print(o) # -> '005'
```

#### `str.find(str)`
Il metodo `find()` restituisce l’index della stringa o carattere ricercato in `str`
```python
s = 'ciao'
print(s.find('a')) # -> 2
```

#### `str.count(str)`
Il metodo `count()` restituisce quante volte `str` è contenuto nella stringa scelta
```python
s = 'pippo'
print(s.count('p')) # -> 3
```

#### `isdecimal() isalnnum() isnumeric() isalpha()`
|                 | `isalpha()` | `isnumeric()` | `isdecimal()` | `isalnum()` |
| --------------- | ----------- | ------------- | ------------- | ----------- |
| abcdef123       | `False`     | `False`       | `False`       | `True`      |
| ??<ABCD!@#$%^&* | `False`     | `False`       | `False`       | `False`     |
| abcWWWXX        | `True`      | `False`       | `False`       | `True`      |
| 01010002        | `False`     | `True`        | `True`        | `True`      |

#### `str.join(list)`
Il metodo `join()` restituisce una stringa formata dalla concatenazione di `list` i cui item sono separati da `str`
```python
L = ['1', '2', '3', '4', '5', '6', '7', '8']ù
''.join(L) # -> '12345678'
```

#### `str.encode(default='utf8')`
La funzione `encode()` prendere in input simboli e legge bytes (può funzionare anche come `ord()`)
```python
out = 'a'.encode()
# faccio encoding del simbolo 'a'
print(out, type(out), sep='  ‐  ') # -> b'a'  ‐  <class 'bytes'>
print('4532[foobar+èù©'.encode()) # -> b'4532[foobar+\xc3\xa8\xc3\xb9\xc2\xa9'
```

#### `str.decode(default='utf8')`
La funzione `encode()` prendere in input byte e fa il decoding in simboli
```python
print('\U0001F923') # -> 🤣
```