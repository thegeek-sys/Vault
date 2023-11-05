---
Created: 2023-10-06
Programming language: "[[Python]]"
Related:
  - "[[for]]"
  - "[[Logical operators]]"
Completed:
---
---
## Introduction
Tramite gli  `if` statement posso utilizzare gli operatori di confronto per far compiere decisioni dal calcolatore. Se la condizione booleana è vera (`True`) eseguirà del codice e se è falsa (`False`) di saltarlo per eseguire le istruzioni all’interno dell’`else` (se presente).

```python
def print_on_even(x):
	if x % 2 == 0:
		print(x, 'è pari')
	else:
		print(x, 'è dispari')

	print_on_even(10) # -> 10 è pari
	print_on_even(7) # -> 7 è dispari
```

Posso anche utilizzare gli operatori logici per concatenare più condizioni insieme, in modo tale di evitare annidamenti che rendono il codice difficilmente leggibile

---
## `elif`
Posso differenziare maggiormente i dati in input attraverso lo statement `elif` (simile allo `switch case` degli altri linguaggi di programmazione)

```python
def confronto(x):
	if x > 0:
		print(x, 'è maggiore di zero')
	elif x < 0:
		print(x, 'è minore di zero')
	else:
		print('x è uguale a zero')

for i in range(0,10):
	confronto(i)
```

---
## Operatore ternario
L’operatore ternario ci permettere di scrivere un `if-else` su una sola riga di codice. La sua sintassi è: `<valore_se_vero> if <condizione> else <valore_se_falso>`

```python
is_nice = True
state = "nice" if is_nice else "not nice"
```