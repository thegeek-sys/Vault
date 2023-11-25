---
Created: 2023-11-06
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Index

1. [[#Formare un percorso assoluto|Formare un percorso assoluto]]
2. [[#Aprire un file|Aprire un file]]
3. [[#Leggere un file|Leggere un file]]
4. [[#Scrivere in un file|Scrivere in un file]]
5. [[#Context manager|Context manager]]
---
## Formare un percorso assoluto
- Usare il + della concatenazione delle stringhe
- Usare il modulo `os` con il comando `os.path.join(str,str)`
- Usare la `f-string`

```python
abs_path = prefix+'/'+rel_path

import os
abs_path = os.path.join(prefix, rel_path)

abs_path = f'{prefix}/{rel_path}' # tramite la f-string posso inserire automaticamente dentro la stringa stessa delle variabili
```

---
## Aprire un file
Quando si apre un file tramite il comando `open()` questo avrà `type` `TextIOWrapper` e bisogna intenderlo come uno stream.

| Character | Meaning                                                         |
| --------- | --------------------------------------------------------------- |
| ‘r’       | open for reading (default)                                      |
| ’w‘       | open for writing, truncating the file first                     |
| 'x'       | create a new file and open it for writing                       |
| ’a‘       | open for writing, appending to the end of the file if it exists |
| ’b‘       | binary mode                                                     |
| ’t‘       | text mode (default)                                             |
| ’+‘       | open a disk file for updating (reading and writing)             |

Il metodo `open()` prende in input:
- **path del file** 
	- percorso assoluto → ovvero a partire dalla root directory
	- path relativo → che funziona solo se ci si trova nella cartella corretta
- modalità di apertura (default: r → read)
- encoding del file (default: utf8)

>[!NOTE] Ricorda
>Ricorda di chiudere il file con il comando `close()`, se non lo faccio non mi salverà un eventuale file modificato o posso incorrere in problemi di gestione della memoria

Il tipo aperto da open è un `TextIOWrapper` che è un generatore che restituisce un riferimento puntatore all’inizio del file. Ogni volta vogliamo leggere una riga dobbiamo pensare ad una testina che carica la riga in Python e che poi passa alla successiva.

```python
fr = open('example.txt', mode='rt', encoding='utf-8')
fw = open('example.txt', mode='wt', encoding='utf-8')
print(type(fr)) # -> <class '_io.TextIOWrapper'>
```

---
## Leggere un file
#### `file.read()`
Il metodo `read()` ci permette di leggere un file in un colpo solo, e farà spostare la testina alla fine del file (non lo potremmo leggere di nuovo se non riaprendo il file o spostando la testina). N.B. Un file per essere letto deve essere aperto in modalità `r`.
```python
fr.read()
fr.read.split('\n') # creo una lista separata in ogni riga
```

#### `file.seek(int)`
Lo utilizzerò per far spostare la testina nel file
- **0:** sets the reference point at the beginning of the file
- **1:** sets the reference point at the current file position
- **2:** sets the reference point at the end of the file

#### `file.tell()`
Il metodo `tell()` ci restituisce il byte (carattere) a cui la testina si trova

#### `file.readline()`
Il metodo `readline()` ci permette di leggere la prima riga di un file

#### `file.readlines()`
Il metodo `readline()` ci restituisce una lista composta da ogni riga del file (N.B. questo metodo mantiene gli `\n` alla fine di ogni riga)

#### `for`
Se non mi interessa avere a disposizione subito il contenuto del file possiamo analizzare un file come un vero e proprio generatore attraverso un `for`
```python
for i, line in enumerate(fr):
    print(line)
    if i == 9:
        break
```

---
## Scrivere in un file
#### `file.write()`
Il metodo `write()` ci permette di scrivere in un file (sempre e solo se è stato aperto in modalità `w`) e usando `\n` per andare a capo.
```python
fw.write("pippo è andato al mare\n")
```

#### `print()`
Posso usare una sintassi particolare della funzione `print()` per scrivere all’interno di un file
```python
print("kjglahgkjhkj kajhhk gj", file=fw) # non mi serve mettere \n
										 # alla fine della stringa
L = ['iacopo','masi','12345']
print(L[0],L[1],L[2], file=fw, sep='\n', end='')
# con n argomenti possiamo passare a print un argomento variabile
print(*L, file=F, sep='\n',end='') # spacchetta gli elementi di L e
								   # li scrive dentro il file
								   # (letteramente fa un print nel
								   # file)
```

---
## Context manager
Il **Context manager** permette di aprire un file senza dover alla fine usare il comando `close()`, ci assicura inoltre che se il codice dentro il context manager genera un errore, questo riuscirà comunque a salvare correttamente il file (o almeno una sua porzione). Il context manager fa uso dello statement `with`.
Il file potrà essere analizzato solo all’interno del context manager stesso.

```python
with open('example.txt', mode='rt') as fr:
	pass

# posso anche aprire più file insieme
with open('01.txt') as f1, open('02.txt') as f2:
	pass
```