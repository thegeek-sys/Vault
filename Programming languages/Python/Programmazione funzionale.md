---
Created: 2023-11-15
Programming language: "[[Python]]"
Related:
  - "[[map()]]"
Completed:
---
---
## Introduction
La **programmazione funzionale**, a differenza della programmazione strutturale, decompone il problema in un **insieme di funzioni** (queste sono dette *purely functional*).
Essendo *purely functional* non avrà side effects infatti qui non verranno usate strutture dati che vengono aggiornate via via che il programma esegue e l’output della funzione dovrà dipendere solamente dall’input (linguaggi Haskell, Lisp sono propriamente purely functional, Python non lo è).

---
## Iterator, Iterable, Iteration (Eager)
In Python esistono due metodi di iteazione:
- **eager** → iteratori
- **lazy** → generatori

### Eager

```start-multi-column
ID: ID_kko9
Number of Columns: 2
Largest Column: standard
border: on
Column Spacing: 10px
```

##### Iterator
E’ un oggetto che implementa un protocollo di iterazione per iterare su uno stream finito/infinito di dati
1. `__iter__()`
2. `__next__()`
3. `RaiseStopIteration`

--- column-end ---

##### Iteratable
Ogni oggetto che può essere usato con un ciclo for (loop over it). Un oggetto è `iterable` se vi fornisce un `iterator` come output.
1. `__iter__()`
2. `__getitem__()`

--- end-multi-column
```python
l = [1,2]
iter_l = iter(lista)
next(iter_l) # -> 1
next(iter_l) # -> 2
next(iter_l) # -> StopIteration
```

### Lazy
La lazy evaluation è potente perché fornisce le iterazioni «on the fly» riuscendo anche ad essere meno memory intensive (la differenza si vede su insiemi di grandi dati) però i generatori possono essere valutati solo una volta

All’interno dei generatori viene utilizzata la keyword `yield`. Informalmente `yield` è simile ad un `return` ma <u>mantiene uno stato</u>. Quando infatti si raggiunge uno `yield`, viene sospeso lo stato di esecuzione del generatore e le variabili locali sono salvate

```python
''' programmazione strutturale '''
def fibon(n):
	f_n = 0
	f_n1 = 1
	result = [f_n, f_n1]
	for i in range(n-2):
		f_n, f_n1 = f_n1, f_n + f_n1
		result.append(f_n1)
	return result
rez = fibon(50)
len(rez),rez[-1] # -> (50, 777872049)

''' programmazione funzionale '''
def fibon(n):
	f_n = 0
	print('iter 0: ', end='')
	yield f_n # "cede" 0 mamantiene lo stato della funzione
	
	f_n1 = 1
	print('iter 1: ', end='')
	yield f_n1 # "cede" 1 mamantiene lo stato della funzione

	for i in range(n-2):
		f_n, f_n1 = f_n1, f_n + f_n1
		print(f'iter {i+2}: ', end='')
		yield f_n1 # "cede" f_n1 mamantiene lo stato della funzione
	# implicitamente raise StopIteration
rez_gen = fibon_gen(50)
type(rez_gen) # -> generator
next(rez_gen) # iter 0: 0
next(rez_gen) # iter 1: 1
next(rez_gen) # iter 2: 2
next(rez_gen) # iter 3: 3
next(rez_gen) # iter 4: 5
```

