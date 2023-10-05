---
Created: 2023-10-05
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Le funzioni possono essere produttive (ritornano un parametro) ma anche non produttive (quando ad esempio al loro interno presentano un print) e, in questo caso, assegnando una variabile ad essa questa ci restituità il tipo `None` (questo avviene spesso, se non si fa attenzione, nelle funzioni ricorsive)

```python
def print_arg(arg):
	print(arg)

output = print_arg('ciao')
print(output) # -> None
type(output) # -> NoneType
```

`None` è dunque una variabile che è definita ma a cui non è stato associato un valore