---
Created: 2025-03-07
Class: 
Related: 
Completed:
---
---
## Introduction
L’organizzazione di un area di memoria è basata sul concetto di file e di directory
Una directory contiene file e directory ed ha una struttura gerarchica da albero (ma solo le directory possono avere figli)

I file regolari contengono sequenze di bit dell’area di memoria sulla quale c’è il filesystem (ASCII o binari), ma esistono file **non regolari** o speciali

Linux ha un solo filesystem principale che ha come directory radice `/` (root). Tutti i file e directory sono contenuti direttamente o indirettamente in `/`
In una stessa directory è vietato creare:
- 2 file con lo stesso nome
- 2 directory con lo stesso nome
- Un file ed una directory con lo stesso nome

---
## Path
Ogni file o directory è raggiungibile dalla directory principale `/` mediante un **path assoluto**, ovvero una sequenza di directory separate da `/`

```bash
/home/utente1/dir1/dir11/dir112/file.pdf
```

>[!info]
>Un’eccezione di path assoluto riguarda l’utente corrente (`userX`), infatti `~` equivale a `/home/userX`

### Current working directory `cwd`
Per conoscere la current working directory si usa il comando `pwd`. Per cambiare cwd usare il comando `cd [path]` dove `[path]` può essere assoluto o relativo (`cd` senza path ritorna alla home)
E’ possibile inoltre usare `..` (parent directory) e `.` (directory attuale)

---
## Contenuto di una directory
Il comando `ls` restituisce la lista dei file contenuti in una directory. Però in una directory ci sono dei file “nascosti” (tipicamente file di configurazione o sile usati a supporto di comandi e applicazioni, es. `.bash_history`); questi sono visibili attraverso `ls [-a | --all]`

Se vogliamo ricorsivamente visualizzare il contenuto delle sotto directory abbiamo l’opzione `[-R | --recursive]`

Si puo’ visualizzare l’albero delle directory con il comando
```bash
tree [-a] [-L maxdepth] [-d] [-x] [nomedir]
```
