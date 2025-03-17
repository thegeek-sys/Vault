---
Created: 2025-03-17
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
In Linux le due entità fondamentali sono:
- **file** → descrivono/rappresentano le risorse
- **processi** → permettono di elaborare dati e usare le risorse

Un file eseguibile, in esecuzione è chiamato **processo**. Per lanciare un processo bisogna eseguire il file corrispondente

>[!example]
>Esempi di processi sono quelli creati eseguendo i comandi delle lezioni precedenti (es. `dd`, `ls`, `cat`, …)

Però non tutti i comandi shell creano dei processi. Ad esempio `echo` e `cd` vengono esguito all’interno del processo di shell

Un file eseguibile più essere eseguito più volte dando vita ad un nuovo processo ogni volta e non è necessario attendere il termine della prima esecuzione per avviare la seconda (Linux è multi-processo)

### Ridirezione dell’output
I simboli `>` e `<` possono essere utilizzati per redirigere l’output di un comando su di un file
Ad esempio:
- `ls>dirlist` → output di `ls` redirezionato in `dirlist`
- `ls>dirlist 2>&1` → l'output di `ls` viene redirezionato in `dirlist`, includendo sia l'output normale (stdout, 1) che gli errori (stderr, 2). Redireziona stderr a stdout che è gia stato redirezionato a `dirlist`, allora anche gli errori finiranno lì
- `ls 2>&1 > dirlist` → redirezione stderr a stdout (il terminale) così che gli errori mi vengano mostrati a terminale e poi redireziono stdout a `dirlist` (solo stdout va nel file, mentre gli errori rimangono sul terminale)

---
## Rappresentazione dei processi
I processi sono identificati da:
- **Process Identifier** (*PID*)
- **Process Control Block** (*PCB*)
- Sei aree di memoria

### Process Identifier
E’ un identificatore univoco di un processo. In un dato istante, non ci possono essere 2 processi con lo stesso PID
Una volta che un processo e’ terminato, il suo PID viene liberato, e potrebbe essere prima o poi riusato per un altro processo

### Process Control Block
Il PCB è unico per ogni processo e contiene:
- PID: Process Identifier
- PPID: Parent Process Identifier
- Real UID: Real User Identifier
- Real GID: Real Group ID
- Effective UID: Effective User Identifier (UID assunto dal processo in esecuzione)
- Effective GID: Effective Group ID (come sopra per GID)
- Saved UID: Saved User Identifier (UID avuto prima dell’eseccuzione del SetUID)
- Saved GID: Saved Group Identifier (come supra per GID)
- Current Working Directory: directory di lavoro corrente
- Umask: file mode creation mask
- Nice: priorita statica del processo

### Aree di memoria
Le sei ree di memoria sono:
