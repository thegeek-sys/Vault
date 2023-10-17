---
Created: 2023-10-06
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
La sintassi del ciclo `for` in generale è:

```python
#non confondere in nel for con operatore appartenenza
for <elemento i-esimo> in <iteratore/generatore>:
	# corpo del ciclo for
	# qui si indicano le istruzioni da ripetere
	# che spesso usano <elemento i-esimo>
	pass
```

## `range()`
In generale se si ha bisogno di una sequenza $[1,2,3,\dots,n-1]$ possiamo usare `range(0,n)` o più brevemente `range(n)`.
La funzione `range(start,stop,inc)` fornisce una sequenza di interi:
- da start
- fino a stop **escluso**
- con un incremento di inc

> [!WARNING]
> Questa funzione **enumera** interi non mi rende una lista direttamente
>
> - se start < end → inc > 0 da start voglio andare ad end (da sx a ds)
> - se start > end → inc < 0 da start vado a ritroso a end (da dx a sx)

``` python
for i in range(0,10):
	print_on_even(i)
```
Il ciclo `for` si usa quando si conosce a priori quanti elementi si devono scandire sapendo il numero di iterazioni (ci permette di iterare su dei tipi iterabili). La sua sintassi è

Applicando quanto scritto

```python
for i in range(10)
	print(i, end=' ') # -> '1 2 3 4 5 6 7 8 9'
```

