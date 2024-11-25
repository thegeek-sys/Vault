---
Created: 2024-11-25
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Processi]]"
Completed:
---
---
## Index
- [[#Strutture di controllo del SO|Strutture di controllo del SO]]
	- [[#Strutture di controllo del SO#Tabelle di memoria|Tabelle di memoria]]
	- [[#Strutture di controllo del SO#Tabelle per l’I/O|Tabelle per l’I/O]]
	- [[#Strutture di controllo del SO#Tabelle dei file|Tabelle dei file]]
	- [[#Strutture di controllo del SO#Tabelle dei processi|Tabelle dei processi]]
- [[#Process Control Block|Process Control Block]]
	- [[#Process Control Block#Come si identifica un processo?|Come si identifica un processo?]]
	- [[#Process Control Block#Stato del processore|Stato del processore]]
	- [[#Process Control Block#Informazioni per il controllo del processo|Informazioni per il controllo del processo]]
	- [[#Process Control Block#Extra|Extra]]
---
## Strutture di controllo del SO
Per gestire al meglio le risorse dei processi il SO ha bisogno di strutture di controllo. Per questo motivo il SO costruisce e mantiene una o più tabelle per ogni entità (in cui salva lo stato di ogni processo e risorsa)

![[Screenshot 2024-10-04 alle 12.49.04.png]]
Esistono quindi delle tabelle per la memoria, delle tabelle per i dispositivi e delle tabelle per l’archiviazione di massa e sono contenute all’interno del kernel
All’interno del primary process table sono contenuti i block dei singoli processi. Al loro interno sono contenute solo le informazioni essenziali mentre solamente nel process image è contenuta tutta la memoria necessaria a un processo

### Tabelle di memoria
Le tabelle di memoria (*memory tables*) sono quelle che al giorno d’oggi servono per gestire la memoria virtuale.
Devono quindi permettere allocazione e deallocazione della memoria principale e secondaria e contengono informazioni per gestire la memoria virtuale

### Tabelle per l’I/O
Le tabelle per l’I/O ci dicono per ogni dispositivo  quali sono le caratteristiche di quel dispositivo

### Tabelle dei file
Le tabelle dei file forniscono informazioni sui nomi dei file e dove solo collocati all’interno della memoria di massa in cui sono memorizzati

### Tabelle dei processi
Le tabelle dei processi sono contenuti i dettagli dei processi in modo tale che il SO possa gestirli:
- stato corrente (Ready, Running ecc.)
- identificatore (per poter distinguere i diversi processi)
- locazione in memoria
- ecc.

Tutte queste informazioni sono presenti all’interno del process control block
Mentre nel **process image** è contenuto l’insieme di programma sorgenti, dati (RAM), stack delle chiamate e PCB (kernel). Modificando un registro o una cella di memoria cambia anche l’immagine

---
## Process Control Block
Il **Process Control Block** è la struttura dati più importante di un sistema operativo e per questo **richiede protezioni** non deve infatti esser possibile che un processo ci acceda in quanto una modifica del PCB avrebbe ripercussioni su tutto il SO
Le informazioni in ciascun blocco di controllo possono essere raggruppate in 3 categorie:
- identificazione
- stato del processore
- controllo

![[Screenshot 2024-10-07 alle 13.10.38.png]]

Questi inoltre sono contenuti all’interno della RAM, che all’accensione, il kernel riserva a sé stesso

### Come si identifica un processo?
Ad ogni processo è assegnato un numero identificativo unico: il **PID** (**P**rocess **ID**entifier)
Talmente è importante che molte tabelle del SO che si occupano di tenere traccia di quali processi hanno eseguito una determinata azione, usano direttamente il PID per identificarlo

> [!info] Se un processo viene terminato il suo PID può essere riassegnato

Nel PCB sono dunque contenuti:
- PID
- PPID (Parent PID)
- identificatore dell’utente proprietario

### Stato del processore
Lo stato del processore è dato dai contenuti dei registri del processore stesso e dal PSW (in cui ricordiamo sono contenute le informazioni di stato)

>[!warning] Non confondere con lo stato, o meglio la modalità del processo (ready, blocked, …)

![[Screenshot 2024-10-07 alle 12.57.29.png]]

### Informazioni per il controllo del processo
Anche queste sono contenute del PCB e sono:
- stato del processo
- priorità
- informazioni sullo scheduling (es. per quanto tempo è stato in esecuzione l’ultima volta)
- se il processo è blocked, è riportanto anche l’evento di cui è in attesa

### Extra
Sono anche contenuti nel PCB:
- puntatori  ad altri processi
- eventualmente liste concatenate di processi nei casi in cui siano necessarie (es. code di processi per qualche risorsa)
- ciò che serve per far comunicare vari processi
- permessi speciali
- puntatori ad aree di memoria (indirizzo a cui inizia la process image)
- file aperti e uso di risorse
