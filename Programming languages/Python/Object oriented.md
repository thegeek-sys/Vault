---
Created: 2023-11-15
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
A differenza della programmazione strutturata, in cui i dati e le funzioni sono separati. Nella programmazione **object oriented** i dati diventano un attributo dell’oggetto e le funzioni diventano i metodi associati all’oggetto (i dati e le funzioni sono incapsulati). Invocare un metodo su un oggetto provoca dei side effect (cambiamenti) di stato ai suoi attributi se mutabile, altrimenti crea un nuovo oggetto in uscita dal metodo (se immutabile).

![[oop int.png]]
```python
list.append(L,4)    L.append(5)
# obj oriented      # strutturale

''' programmazione strutturale '''
a = 1
b = 100
def somma(x, y):
	return x + y
somma(a,b) # -> 101

''' programmazione ad oggetti '''
a = int(1)
b = int(100)
int.__add__(a,b) # -> 101
# __add__ è un metodo della classe int a cui passo i parametri a e b
```

---
## Classi e Oggetti

```start-multi-column
ID: ID_lejf
Number of Columns: 2
Column Size: [75%, 25%]
column spacing: 0px
border: off
```

![[oop int.png]]

--- column-end ---

test

--- end-multi-column


ddw