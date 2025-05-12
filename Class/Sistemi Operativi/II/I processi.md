---
Created: 2025-03-17
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
	- [[#Introduction#Ridirezione dell’output|Ridirezione dell’output]]
- [[#Rappresentazione dei processi|Rappresentazione dei processi]]
	- [[#Rappresentazione dei processi#Process Identifier|Process Identifier]]
	- [[#Rappresentazione dei processi#Process Control Block|Process Control Block]]
	- [[#Rappresentazione dei processi#Aree di memoria|Aree di memoria]]
- [[#Stato di un processo|Stato di un processo]]
- [[#Modalità di esecuzione dei processi|Modalità di esecuzione dei processi]]
	- [[#Modalità di esecuzione dei processi#Esecuzione in background|Esecuzione in background]]
	- [[#Modalità di esecuzione dei processi#Lista di job|Lista di job]]
	- [[#Modalità di esecuzione dei processi#Comandi $\verb|bg|$ e $\verb|fg|$|Comandi $\verb|bg|$ e $\verb|fg|$]]
- [[#Pipelining dei comandi|Pipelining dei comandi]]
- [[#$ verb ps [opzioni] [pid...] $|ps]]
- [[#$ verb top [-b] [-n num] [-p {pid}] $|top]]
- [[#$ verb kill [-l [signal [-signal] [pid...] $|kill]]
	- [[#Alcuni segnali|Alcuni segnali]]
	- [[#$ verb SIGUSR1 $ e $ verb SIGUSR2 $|SIGUSR1 e SIGUSR2]]
- [[#$ verb nice [-n num] [command] $|nice]]
- [[#$ verb renice priority {pid} $|renice]]
- [[#$ verb strace [-p pid] [command] [-o file] $|strace]]
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

---
## $\verb|ps [opzioni] [pid...]|$
Il comando `ps` mostra le informazioni riguardo una selezione dei processi attivi (se si vuole un aggiornamento continua della selezione e le informazioni mostrare usare `top`)
Legge le informazioni dai file virtuali in `/proc`

`ps` senza argomenti mostra i processi dell’utente attuale lanciati dalla shell corrente. Per ognuno di essi mostra `PID`, `TTY`, `TIME` (tempo totale di esecuzione) e `CMD`

Vediamo ora le opzioni disponibili:
- `-e` → tutti i processi di tutti gli utenti lanciati da tutte le shell o al boot (figli del processo 0)
- `-u {user}` → tutti i processi degli utenti nella lista in input
- `-p {pid}` → tutti i processi con i PID nella lista
- `-f` → restituisce in output delle colonne addizionali quali `UID`, `PPID`, `C` (fattore di utilizzo della CPU, da $0$ a $99$) e `STIME` (tempo di avvio)
- `-l` → altre colonne addizionali quali `F` (flag), `PRI` (priorità del processo, più basso → più alta priorità), `NI` (nice value, influenza la priorità), `ADDR` (indirizzo di memoria del processo), `SZ` (dimensione dell’immagine del processo in pagine), `WCHAN` (indirizza la funzione in cui il processo è in attesa, se dormiente)
- `-o {fields}` → per scegliere i campi da visualizzare
- `-C {cmds}` → mostra solo i processi il cui nome eseguibile è in `{cmds}`

Ci stanno anche i campi `RUSER` per il reale utente che ha avviato il processo e `EUSER` che corrisponde all’utente che ha eseguito il processo

### Significato campi di output
- `PPID` → parent pid, pid del processo che ha creato questo processo
- `C` → parte intera della percentuale di uso della CPU
- `STIME` (o `START`) → l'ora in cui e stato invocato il comando, oppure la data, se e stato fatto partire da piu di un giorno
- `TIME` → tempo di CPU usato finora
- `CMD` → comando con argomenti
- `F` → flags associati al processo: 1 il processo e stato “forkato", ma ancora non eseguito; 4, ha usato privilegi da superutente; 5, entrambi i precedenti; 0, nessuno dei precedenti (`-y -l` elimina questo campo che e’ poco utile)
- `S` → stato (modalita) del processo in una sola lettera
- `UID` → utente che ha lanciato il processo (se SetUID presente pottrebbe non essere chi ha dato il comando)
- `PRI` → attuale priorita del processo (piu il numero e alto, minore e la priorita)
- `NI` → valore di nice, da aggiungere alla priorita’ (vedere piu avanti)
- `ADDR` → indirizzo in memoria del processo, ma e mostrato (senza valore) solo per compatibilita all'indietro (`-y -l` toglie questo campo e lo sostituisce con RSS - resident set size - dimensione del processo in memoria principale in $KB$ - non tiene conto delle pagine su disco)
- `SZ` → dimensione totale attuale del processo in numero di pagine (tutte le 6 aree di memoria del processo) sia in memoria che su disco (memoria virtuale)
- `WCHAN` → se il processo e in attesa di un qualche segnale o comunque in sleep, qui c'e la funzione del kernel all'interno della quale si e fermato

---
## $\verb|top [-b] [-n num] [-p {pid}]|$
Permette di avere un `ps` ma costantemente aggiornato
- `-b` → non accetta più comandi interattivi, ma continua a fare refresh ogni pochi secondi
- `-n num` → fa solo `num` refresh
- `-p {pid}` → come in `ps`

Una volta aperto in modo interattivo, basta premere `?` per avere la lista dei comandi accettati

---
## $\verb|kill [-l [signal]] [-signal] [pid...]|$
Permette di inviare segnali ad un processo (non sono la terminazione)
- `-l` → mostra la lista dei segnali; un segnale `signal` è identificato da un numero oppure dal nome con `SIG` o senza `SIG` (es. `kill -9 pid` oppure `kill -l SIGKILL pid` oppure `kill -l KILL pid`)

I segnali verranno presi in considerazione solo se il *real user* del processo è lo stesso che invia il segnale (oppure se lo invia un superuser).
Un processo che riceve un segnale fa o un’azione predefinita (`man 7 signal`) o un’azione personalizzata

### Alcuni segnali
- `SIGSTOP`, `SIGSTP` → sospensione processo
- `SIGCONT` → per la continuazione di processi stoppati
- `SIGKILL`, `SIGINT` → per la terminazione dei processi

>[!info]
>- `CTRL+Z` invia un `SIGSTOP`
>- `CTRL+C` invia un `SIGINT`
>- `bg` invia `SIGCONT` al job indicato
>- con `kill` si può usare la notazione con `%` (usata in `bg` e `fg`) per indicare i job destinatari del messaggio

### $\verb|SIGUSR1|$ e $\verb|SIGUSR2|$
I segnali `SIGUSR1` e `SIGUSR2`sono impostai per essere usati dall’utente per le proprie necessità. Essi consentono una semplice forma di comunicazione tra processi

>[!example]
>In un programma $P_{1}$ si puó definire un gestore del segnale (signal handler) per `SIGUSR1`(o 2). Se un programma $P_{2}$ invia un `SIGUSR1` (o 2) a $P_{1}$, $P_{1}$ eseguirà il codice del gestore del segnale

---
## $\verb|nice [-n num] [command]|$
`nice` senza opzioni dice quant’è il *niceness* di partenza. Il niceness può essere pensato come un’addizione sulla priorità: se positivo, ne aumenta il valore (quindi la priorità decresce), altrimenti ne diminuisce il valore (la priorità cresce). Questo può andare a $-19$ a $+20$ con valore di default $0$

`command` → lancia `command` con niceness `num` ($0$ se non dato) 

---
## $\verb|renice priority {pid}|$
Interviene su processi già in esecuzione (infatti richiede dei PID) e serve per cambiare la priorità dei processi

---
## $\verb|strace [-p pid] [command] [-o file]|$
Lancia `command` mostrando tutte le sue system calls, oppure visualizza le system call del processo `pid`. Tramite `-o file` ridimensiona l’output su un file

Ci sarà utile per il debug dei programmi che usano le system call