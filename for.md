---
Created: 2023-10-06
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
I cicli `for` ci permettono di iterare su dei tipi iterabili

```python
def print_on_even(n):
	if n % 2 == 0:
		print(n, 'è pari')
	else:
		pass

for i in range(0,10):
	print_on_even(i)
```
