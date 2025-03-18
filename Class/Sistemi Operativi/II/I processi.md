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
- **Text Segmento** → le istruzioni da eseguire
- **Data Segment** → i dati statici (variabili globali, variabili locali static) inizializzati e alcune costanti di ambiente
- **BSS** → dati statici non inizializzati (block started from symbol); la distinzione dal segmento dati si fa per motivi di realizzazione hardware
- **Heap** → dati dinamici (allocati con malloc e simili)
- **Stack** → chiamate a funzioni, con i corrispondenti dati dinamici (variabili locali non static)
- **Memory Mapping Segment** → tutto ciò che riguarda librerie esterne dinamiche usate dal processo, nonché estensione dello heap in alcuni casi

Alcune aree di memoria però potrebbero essere condivise:
- il text segment tra più istanze dello stesso processo
- 2 processi potrebbero avere lo stesso BSS o Data segment o MMS
- Lo stack non è mai condiviso

![[Pasted image 20250317230250.png|600]]

---
## Stato di un processo
Un processo può trovarsi in vari stati:
- **Running** (R) → in esecuzione su un processore
- **Runnable** (R) → pronto per essere eseguito (non è in attesa di alcun evento); in attesa che lo scheduler lo (ri)seleziona per l’esecuzione
- (Interruptible) **Sleep** (S) → e in attesa di un qualche evento (ad esempio, lettura di blocchi dal disco), e non puo quindi essere scelto dallo scheduler
- **Zombie** (Z) → il processo e’ terminato e le sue 6 aree di memoria non sono più in memoria; tuttavia, il suo PCB viene ancora mantenuto dal kernel perche il processo padre non ha ancora richiesto il suo “exit status" (ritorneremo su questo punto)
- **Stopped** (T) → caso particolare di sleeping: avendo ricevuto un segnale STOP, e in attesa di un segnale CONT
- **Traced** (t) → in esecuzione di debug, oppure in generale in attesa di un segnale (altro caso particolare di sleeping; vedremo più avanti)
- **Uninterruptible sleep** (D) → come sleep, ma tipicamente sta facendo operazioni di IO su dischi lenti e non può essere interrotto ne ucciso

---
## Modalità di esecuzione dei processi
Ci sono due modi per poter eseguire i processi:
- **forground**
- **background**

I **forground** sono praticamente tutti quelli visti fino ad ora, ovvero quelli in cui il comando può leggere l’input da tastiera e scrivere su schermo. Finché questo non termina, il prompt non viene restituito e non si possono sottomettere altri comandi alla shell
I processi in **background** non possono leggere l’input da tastiera ma può scrivere su schermo. In questo caso il prompt viene immediatamente restituito e mentre il job viene eseguito in background, si possono da subito dare altri comandi alla shell

### Esecuzione in background
Per eseguire un comando in background viene usato l’operatore **`&`** (ampersand) che non è disponibile in tutte le shell. Questo viene posto alla fine di un comando

>[!example] `sleep 15s &`

### Lista di job
Per vedere la lista di job in esecuzione si usa il comando `jobs [-l] [-p]`

![[Pasted image 20250317231704.png|500]]
Se un job è composto da più task

### Comandi $\verb|bg|$ e $\verb|fg|$
Il comando `bg` permette di portare un processo in background (lancio il processo; lo interrompo con `CTRL+Z`; lo risveglio con `bg`)
Invece `fg [job_id]` dove `[job_id]` è il numero del job, lo riporta in foreground, mentre `gb [job_id]` porta in background il processo `[job_id]`

![[Pasted image 20250317232711.png|400]]

Si possono identificare i job anche con:
- `[prefix]` → dove `prefix` è la parte iniziale del job desiderato
- `+` oppure `%` → l’ultimo job mandato
- `-` → il penultimo job mandato

---
## Pipelining dei comandi
Per eseguire un job composto da più comandi si usa la **pipe** `|`

>[!example]
>```bash
>comando1 | comando2 | … comando n
>```

**Lo standard output di un comando $i$ diventa l’input del commando $i+1$**

Se uso `|&` è lo standard error che viene redirezionato sullo standard input del comando successivo. Inoltre bisogna ricordare che il comando $i+1$ non deve necessariamente usare lo stdout/stderr del comando $i$
