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