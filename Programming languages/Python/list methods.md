---
Created: 2023-10-12
Programming language: "[[Python]]"
Related:
  - "[[list methods]]"
  - "[[list]]"
Completed:
---
---
#### `list.append()`
Per aggiungere elementi in coda alle liste viene utilizzata il metodo `append()` che modifica la lista **“in-place”** (non la devo riassegnare)
```python
lista = [1,2,3,5]
lista.append(4)
lista += [4] # è òa stessa cosa
print(lista) # -> [1,2,3,5,4]

lista = lista.append(4)
print(lista) # -> None
```

#### `list.extend()`
Il metodo `estend()` serve per aggiungere una lista in coda ad una lista (se provassi a fare la stessa cosa con `append()` mi ritroverò con delle liste concatenate)
```python
t = ['a', 'b', 'c']
s = ['d', 'e']

t.extend(s)
print(t) # -> ['a', 'b', 'c', 'd', 'e']
```

#### `list.remove()`
Il metodo `remove()`serve il primo elemento che corrisponde all’oggetto dato
```python
t = ['a', 'b', 'c', 'd', 'e', 'b']
t.remove('b')
print(t) # -> ['a', 'c', 'd', 'e', 'b']
```

#### `list.pop()`
Il metodo `pop()` mi permette di eliminare l’elemento corrispondente all’indice specificato e eventualmente di poterlo assegnare ad una variabile
```python
t = ['a', 'b', 'c', 'd', 'e']
removed = t.pop(1)
print(t,removed) # -> ['a', 'c', 'd', 'e] b
```

#### `del list[int]`
La funzione `del` eliminerà l’indice `int` della lista data
```python
t = ['a', 'b', 'c', 'd', 'e']
del t[0]
print(t) # -> ['b', 'c', 'd', 'e]
```

#### `list.clear()`
Il metodo `clear()` ci permette di rimuovere tutti gli elementi di una lista in-place

#### `list.count(elem)`
Il metodo `count()` ci restituirà il numero di `elem` presenti nella lista data

#### `list.sort()` e `sorted(list)`
Il metodo `sort()` ci permette di ordinare in modo **stabile** (l’ordine parziale di due elementi uguali è mantenuto) tutti gli elementi di una lista in-place in modo crescente.
La funzione `sorted()` ci permette di creare una copia ordinata di tutti gli elementi di una lista

> [!WARNING]
> Possono essere ordinati solo liste composte da stessi tipi

Entrambe queste metodologie supportando la key-word `reversed=True` (di default=`False`) che ci permette di ordinare una lista in modo decrescente e anche l’**ordinamento tramite chiave parziale**.
L’ordinamento tramite chiave parziale ci permette di ordinare una lista a nostro piacimento basandosi sulle funzioni integrate in Python. Per esempio voglio ordinare una lista per lunghezza degli elementi

```python
L = ['gli', 'eroi', 'son', 'tutti', 'giovani', 'e', 'belli']
LS = sorted(s, key=len) # a parità di lunghezza restituirà gli
						#elementi nello stesso ordine di origine

print(LS) # -> ['e','gli','son','eroi','tutti','belli','giovani']
```

Oltre a fare ciò è anche possibile concatenare diversi criteri di ordinamento. Se per esempio volessimo:
- **1° criterio**꞉ Ordinami le stringhe in ordine crescente per LUNGHEZZA
- **2° criterio**꞉ A parità di LUNGHEZZA, ordinale in modo alfabetico

```python
L = ['gli', 'eroi', 'son', 'tutti', 'giovani', 'e', 'belli']

def len_and_value(elem):
	return len(elem), elem

LS = sorted(L, key=len_and_value)
print(LS) # -> ['e','gli','son','eroi','belli','tutti','giovani']
		  #     ha scambiato 'tutti' con 'belli'
```
##### Esempi
**Problema: ordinare una lista per lunghezza delle stringhe, in caso si parità di lunghezza, in ordine INVERSO lessicografico**
```python
def len_inv_less(elem):
	return -len(elem), elem

L = ['gli', 'eroi', 'son', 'tutti', 'giovani', 'e', 'belli']
LS = sorted(L, reverse=True, key=len_inv_less)
lenght = []
for i in LS:
	lenght.append(len_inv_less(i))

print(LS, lenght, sep='\n')
# -> ['e', 'son', 'gli', 'eroi', 'tutti', 'belli', 'giovani']
# -> [(-1, 'e'), (-3, 'son'), (-3, 'gli'), (-4, 'eroi'), (-5, 'tutti'), (-5, 'belli'), (-7, 'giovani')]
```

**Problema: ordinare prima pari poi dispari, a parità di pari per valore**
```python
def pari(elem):
	p = elem%2
	return p, elem

lista = [5,7,4,2,100,11,200]
sorted(lista, key=pari) # -> [2, 4, 100, 200, 5, 7, 11]
```
#### `lista.reverse()` e `reversed(list)`
Il metodo `reverse()` ci permette di invertire l’ordine di tutti gli elementi di una lista in-place.
La funzione `reversed()` ci permette di creare una copia ordinata al contrario di tutti gli elementi di una lista.
