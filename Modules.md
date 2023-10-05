---
Created: 2023-10-05
Programming language: "[[Python]]"
Related:
  - "[[Functions]]"
  - "[[Special Variables]]"
Completed:
---
---
## Introduction
I **moduli** sono dei particolari tipi in python che ci permettono di importare funzioni da altri file.

```python
''' l03.py '''
def create_table(row, col):
	hed = "| hed "*col+'\n'
	sep = "| --- "*col+'\n'
	val = ("| val "*col+'\n')*row
	table = hed+sep+val
	return table

if __name__ == "__main__":
	# se il programma è eseguito NON come modulo ma come CLIENT allora
	final = crate_table(3,8)
	print(final)


''' main.py '''
import l03
type(l03) # -> <class 'module'>
dir(l03) # mi mostrerà solamente la funzione stessa ma non la variabile all'interno dell'if statement
```

Come si può notare nell’esempio precedente è stato usata la variabile `__name__`![[Special Variables#`__name__`]]
Per poter capire se posso importare un file all’interno del mio programma è controllare se il path del modulo desiderato si trova in `sys.path`
```python
import sys
print(sys.path)
```
