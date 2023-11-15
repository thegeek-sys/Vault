---
Created: 2023-11-15
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
La **programmazione funzionale**, a differenza della programmazione strutturale, decompone il problema in un **insieme di funzioni** (queste sono dette *purely functional*).
Essendo *purely functional* non avrà side effects infatti qui non verranno usate strutture dati che vengono aggiornate via via che il programma esegue e l’output della funzione dovrà dipendere solamente dall’input (linguaggi Haskell, Lisp sono propriamente purely functional, Python non lo è).

---
## Iterator, Iterable, Iteration (Eager)
In Python esistono due metodi di iteazione:
- **eager** → iteratori (visti fin’ora)
- **lazy** → generatori


```start-multi-column
ID: ID_kko9
Number of Columns: 2
Largest Column: standard
border: off
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

