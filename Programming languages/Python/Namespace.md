---
Created: 
Programming language: "[[Python]]"
Related:
  - "[[Functions]]"
Completed:
---
---
## Introduction
Per differenziare tutte le variabili che possono venir scritte in un codice complesso, l'interprete Python mantiene un **namespace** separato per ciascuna sede

```python
x = 'python'
def funct():
	x = 'java'
	print(x) # -> 'java'

print(x) # -> 'python'
```

Come possiamo notare ha più "importanza" la variabile definita all'inizio. La variabile `x` definita esternamente infatti fa parte del namespace `global` mentre le variabili definite internamente ad ogni funzioni hanno **ciascuna un proprio namespace** `local`. Per vederle possiamo utilizzare i comandi `globals()` e `locals()`

```python
x = 'main'

def foobar(p):
    x = 'pippo'
    return locals()

def func(a, b, c):
    return locals()

print(globals()) # elenco di tutte le globals
print(foobar(2)) # -> {'p': 2, 'x': 'pippo'}
print(func(1, 2, 3)) # -> {'a': 1, 'b': 2, 'c': 3}
```
## Gerarchia Namespace
Considerando la var **x**, l'ordine è:
1. **Locale** → Se siamo dentro una funzione, l'interprete prima cerca nello scope piu interno quindi dentro la funzione.
2. **Enclosing** → Se **x** non è nello scopo locale ma la funzione corrente è annidata in un'altra funzione, allora si cerca dentro lo scope della funzione "sopra".
3. **Global**: Altrimenti cerca la variabile nello scope globale
4. **Built-in** → Se ancora non trova niente prova nello scope delle funzioni built-int di python.
5. Solleva un errore di tipo `NameError` perché non sa risolvere il nome associato alla variabile
