---
Created: 2023-10-05
Programming language: "[[Python]]"
Related: "[[str]]"
Completed: true
---
## Introduction
Le stringhe, essendo più complesse, possono essere interpretate come un vettore, dunque:

|char.|f|l|a|v|i|o|
|--|--|--|--|--|--|--|
|idx.|0|1|2|3|4|5|
Attraverso le `[]` posso iterare all'interno della stringa

> [!warning]
> l'iterazione della stringa inizia da 0 ma la lunghezza è data dal numero di char

```python
nome = 'flavio' print(nome[0]) # -> 'f'
print(len(nome)) # -> 6 
```
## Usecase
Posso usare `[srt:end:step]` per fare lo slice all'interno di una stringa. Se non metto lo `step` questo sarà uguale a 1 ma se lo configuro negativo questo conterà in direzione contraria

```python
nome = 'iacopo masi' # adesso estraggo copo
print(nome[2:6]) # -> 'copo'

## da notare come il char 6 sia escluso ##
```

Per cercare una sottostringa in una stringa posso usare la funzione `index` che mi darà come output il vettore del carattere di inizio di una determinata stringa

```python
nome = 'iacopo masi'
start = nome.index('copo')
print(start) # -> '2'
print(nome[start:start+len('copo')] # [2,2+4] -> 'copo'
```

Per tagliare tot caratteri devo usare `[idx:]` e al contrario `[:idx]`

```python
# tutti i caratteri fino all'index 2 escluso
print(nome[2:]) # -> 'copo masi'

# tutti i caratteri prima dell'index 3 escluso
print(nome[:3]) # -> 'iac'
```

Python può inoltre, a differenza di molti linguaggi di basso livello, invertire la sequenza di lettura

```python
''' 
 +---+---+---+---+---+---+ 
 | P | y | t | h | o | n | 
 +---+---+---+---+---+---+ 
 0   1   2   3   4   5   6
-6  -5  -4  -3  -2  -1 
'''
nome = 'Python'
print(nome[-1]) # -> 'n'
print(nome[4:3]) # srt >= end -> stringa vuota
print(nome[-4:-1]) # -> 'tho'
nome[0:6:2] # -> "Pto"
nome[4:3:-1] # -> "o"
nome[-3:-4:-1] # -> "h"
nome[-1:-7:-1] # -> "nohtyP"
''' posso anche evitare di mettere start (sottointeso 0) e end (sottointeso 6) '''
print(nome[::-1]) # -> 'nohtyP'

''' stampando da 2 a -1 mi mostrerà il corrispondente in 5 ''' nome[2:-1] # OUT -> "tho"
```
