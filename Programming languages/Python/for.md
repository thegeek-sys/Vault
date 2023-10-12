---
Created: 2023-10-06
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
```python
def print_on_even(n):
	if n % 2 == 0:
		print(n, 'è pari')
	else:
		pass

for i in range(0,10):
	print_on_even(i)
```
Il ciclo `for` si usa quando si conosce a priori quanti elementi si devono scandire sapendo il numero di iterazioni (ci permette di iterare su dei tipi iterabili). La sua sintassi è

```python
#non confondere in nel for con operatore appartenenza
for <elemento i-esimo> in <iteratore/generatore>:
	# corpo del ciclo for
	# qui si indicano le istruzioni da ripetere
	# che spesso usano <elemento i-esimo>
	pass
```

Applicando quanto scritto

```python
for i in range(10)
	print(i, end=' ') # -> '1 2 3 4 5 6 7 8 9'
```

