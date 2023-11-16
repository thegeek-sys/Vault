---
Created: 2023-11-16
Programming language: "[[Python]]"
Related:
  - "[[map()]]"
  - "[[Programmazione funzionale]]"
Completed:
---

---
## Introduction
La funzione `map()` permette di “mappare” una funzione f() element-wise ad un iterable. Ritorna infatti un iteratore sulla sequenza data

$$
[a_{1},a_{2},\dots,a_{n}] \rightarrow [f(a_{1}),f(a_{2},\dots,f(a_{n}))]
$$
```python
l = ['sentence', 'fragment']

[upper(s) for s in l] # -> ['SENTENCE', 'FRAGMENT']
list(map(upper, l)) # -> ['SENTENCE', 'FRAGMENT']
```

Nella programmazione funzionale, impariamo a ragionare solamente sull’i-th elemento.
```python
def func(item_a, item_b):
	return item_a.upper(), item_b.lower()

                   # iterable a       # iterable b
mapped = map(func, ['iAcoPo','leTi'], ['maSi','tincoLINI'])

for x, y in mapped:
	print(x,y)

# -> IACOPO masi
# -> LETI tincolini
```