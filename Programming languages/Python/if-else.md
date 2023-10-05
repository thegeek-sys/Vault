---
Created: 2023-10-06
Programming language: "[[Python]]"
Related:
  - "[[for]]"
Completed:
---
---
## Introduction
Utilizzando gli  `if` statement posso utilizzare gli operatori di confronto che ci permette di eseguire del codice se il confronto risulta `True` e di saltarlo se invece il confronto è `False`

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