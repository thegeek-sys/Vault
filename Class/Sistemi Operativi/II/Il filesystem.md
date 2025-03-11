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

---
## Creazione di directory e file
Per creare directory:
```bash
mkdir [-p] nomedir
```
Crea la directory `nomedir` vuota. Se si vuole creare un intero path di directory (es. `./dir11/dir12/dir13`) è necessario l’opzione `-p`

Per creare un file:
```bash
touch nomefile
```
Crea il file `nomefile` vuoto

---
## root directory
Il filesystem root (`/`) contiene elementi eterogenei:
- disco interno solido o magnetico
- filesystem su disco esterno (es. usb)
- filesystem di rete
- filesystem virtuali (usati dal kernel per gestire risorse)
- filesystem in memoria principale

Tutto questo è possibile solo grazie al meccanismo del *mounting*

### mouting
Una qualsiasi directory $D$ dell’albero gerarchico può diventare punto di mount per un altro filesystem $F$ se e solo se la directory root di $F$ diventa accessibile da $D$.

Si hanno due possibilità per $D$:
- se $D$ è vuota, dopo il mount conterrà $F$
- se $D$ non è vuota, dopo il mount conterrà $F$ ma ciò non significa che i dati che vi erano dentro sono persi, saranno infatti **di nuovo disponibili dopo l’unmount** di $F$

---
## Partizioni e mounting
Un singolo disco può essere suddiviso in due o più partizioni.
Una partizione $A$ può contenere il sistema operativo e la partizione $B$ i dati degli utenti (home directory degli utenti). La partizione $A$ verrà montata su `/`, mentre la partizione $B$ verrà montata su `/home`

