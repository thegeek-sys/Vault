---
Created: 2023-10-06
Programming language: "[[Python]]"
Related:
  - "[[for]]"
Completed:
---
---
## Introduction
Tramite gli  `if` statement posso utilizzare gli operatori di confronto per far compiere decisioni dal calcolatore. Se la condizione booleana è vera (`True`) eseguirà del codice e se è falsa (`False`) di saltarlo per eseguire le istruzioni all’interno dell’`else` (se presente).

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