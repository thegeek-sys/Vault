---
Created: 2023-11-06
Programming language: "[[Python]]"
Related: 
Completed:
---
---
#### Introduction
Per aprire un file in Python dobbiamo fornire:
- **path del file** 
	- percorso assoluto → ovvero a partire dalla root directory
	- path relativo → che funziona solo se ci si trova nella cartella corretta
- modalità di apertura (`rt` → read, testuale)
- encoding del file
Quando si apre un file questo avrà `type` `TextIOWrapper` e bisogna intenderlo come uno stream in cui vi è una testina virtuale che arriva in fondo al file quando si da il comando `read()`. È per questo che per leggere nuovamente il file lo dovremmo chiudere e riaprire

```python
file = open('profilazione.py', mode='rt', encoding='utf-8')
print(type(file)) # -> <class '_io.TextIOWrapper'>
file.read()
file.close()
```

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
Il tipo aperto da open è un `TextIOWrapper` che è un generatore che restituisce un riferimento puntatore all’inizio del file. Ogni volta vogliamo leggere una riga dobbiamo pensare ad una testina che carica la riga in Python e che poi passa alla successiva.

```python
fr = open('example.txt', mode='rt', encoding='utf-8')
print(type(fr)) # -> <class '_io.TextIOWrapper'>
```
---
## Leggere un file
#### `file.read()`
Il metodo `read()` ci permette di leggere un file in un colpo solo, e farà spostare la testina alla fine del file (non lo potremmo leggere di nuovo, se non riaprendo il file)
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