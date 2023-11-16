---
Created: 2023-11-16
Programming language: "[[Python]]"
Related:
  - "[[Programmazione funzionale]]"
Completed:
---
---
## Introduction
Possiamo considerare `any()` e `all()` come la versione funzionale rispettivamente di `or` e `and`.
La funzione `any()` infatti ritorna `True` se un qualsiasi elemento di un iterable è `True`, altrimenti ritorna `False`
La funzione `all()` ritorna `True` se tutti gli elementi di un iterable sono `True`

```python
def func(item_a, item_b):
	return item_a.upper(), item_b.lower()

mapped = map(func,['iAcoPo','leTi'],['maSi','tincoLINI'])
mapped_bool = map(lambda xy: xy[0] == 'IACOPO' and xy[1] == 'masi',
				  mapped)
'si esiste almeno uno iacopo masi!' if any(mapped_bool) else 'No'
# -> 'si esiste almeno un masi!'

'si chiamano tutti iacopo masi' if all(mapped_bool) else 'esiste almeno un nome diverso da iacopo masi'
# -> 'esiste almeno un nome diverso da iacopo masi'
```

