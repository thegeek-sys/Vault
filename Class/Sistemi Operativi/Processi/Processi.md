---
Created: 2024-10-03
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Index
- [[#Requisiti di un SO|Requisiti di un SO]]
- [[#Cos’è un processo?|Cos’è un processo?]]
- [[#Elementi di un processo|Elementi di un processo]]
	- [[#Elementi di un processo#Process Control Block|Process Control Block]]
- [[#Traccia di un processo|Traccia di un processo]]
	- [[#Traccia di un processo#Esempio|Esempio]]
- [[#Modello dei processi a 2 stati|Modello dei processi a 2 stati]]
- [[#Creazione e terminazione di processi|Creazione e terminazione di processi]]
	- [[#Creazione e terminazione di processi#Creazione|Creazione]]
	- [[#Creazione e terminazione di processi#Terminazione|Terminazione]]
		- [[#Terminazione#Normale completamento|Normale completamento]]
		- [[#Terminazione#Uccisioni|Uccisioni]]
- [[#Modello dei processi a 5 stati|Modello dei processi a 5 stati]]
- [[#Processi sospesi|Processi sospesi]]
	- [[#Processi sospesi#Motivi per sospendere un processo|Motivi per sospendere un processo]]
- [[#Processi e Risorse|Processi e Risorse]]
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
- [[#Modalità di esecuzione|Modalità di esecuzione]]
	- [[#Modalità di esecuzione#Kernel Mode|Kernel Mode]]
	- [[#Modalità di esecuzione#Da User Mode a Kernel Mode e ritorno|Da User Mode a Kernel Mode e ritorno]]
	- [[#Modalità di esecuzione#System call sui Pentium|System call sui Pentium]]
- [[#Creazione di un processo|Creazione di un processo]]
- [[#Switching tra processi|Switching tra processi]]
	- [[#Switching tra processi#Quando effettuare uno switch?|Quando effettuare uno switch?]]
	- [[#Switching tra processi#Passaggi per lo switch|Passaggi per lo switch]]
	- [[#Switching tra processi#Il SO è un processo?|Il SO è un processo?]]
		- [[#Il SO è un processo?#Il Kernel non è un processo|Il Kernel non è un processo]]
		- [[#Il SO è un processo?#Esecuzione all’interno dei processi utente|Esecuzione all’interno dei processi utente]]
		- [[#Il SO è un processo?#SO è basato sui processi|SO è basato sui processi]]
	- [[#Switching tra processi#Caso concreto: Linux|Caso concreto: Linux]]
- [[#Unix Release 4|Unix Release 4]]
	- [[#Unix Release 4#Transizioni tra stati dei processi|Transizioni tra stati dei processi]]
	- [[#Unix Release 4#Processo Unix|Processo Unix]]
		- [[#Processo Unix#Livello utente|Livello utente]]
		- [[#Processo Unix#Livello registro|Livello registro]]
		- [[#Processo Unix#Livello sistema|Livello sistema]]
	- [[#Unix Release 4#Process Table Entry|Process Table Entry]]
	- [[#Unix Release 4#U-Area|U-Area]]
	- [[#Unix Release 4#Creazione di un processo in Unix|Creazione di un processo in Unix]]
- [[#Thread|Thread]]
	- [[#Thread#ULT vs. KLT|ULT vs. KLT]]
- [[#Linux|Linux]]
	- [[#Linux#Processi e thread|Processi e thread]]
	- [[#Linux#Stati dei processi|Stati dei processi]]
	- [[#Linux#Processi parenti|Processi parenti]]
	- [[#Linux#Segnali ed interrupt|Segnali ed interrupt]]
---

---

---

---


---
## Unix Release 4
Utilizza la seconda opzione in cui la maggior parte del SO viene eseguito all’interno dei processi utente in modalità kernel

### Transizioni tra stati dei processi
![[Screenshot 2024-10-08 alle 00.09.44.png]]
Dal kernel running si può passare a preempted, ovvero quel momento prima che finisca il processo in cui il kernel per qualche motivo decide di togliergli il processore.
Quando un processo finisce, prima che muoia, va nello stato zombie, in cui tutta la memoria di quel processo viene deallocata (compresa l’immagine) e l’unica cosa che sopravvive è il process control block con l’unico scopo di comunicare l’exit status al padre; una volta che il padre ha ricevuto che il figlio gli ha dato questo exit status, a quel punto anche il PCB viene tolto e il processo figlio viene definitivamente terminato
Da notare che un processo in kernel mode non è interrompibile che non lo rendeva adatto ai processi real-time

In sintesi
**User running** → in esecuzione in modalità utente; per passare in questo stato bisogna necessariamente passare per kernel running in quanto è avvenuto un process switch, l’unica cosa che può avvenire è tornare in kernel running in seguito ad una system call o interrupt
**Kernel running** → in esecuzione in modalità kernel o sistema
**Ready to Run, in Memory** → può andare in esecuzione non appena il kernel lo seleziona
**Asleep in Memory** → non può essere eseguito finché un qualche evento non si manifesta e ci è diventato a seguito di un evento bloccante; il processo è in memoria, corrisponde al blocked del modello a 7 stati
**Ready to Run, Swapped** → può andare in esecuzione (non è in attesa di eventi esterni), ma prima dovrà essere portato in memoria
**Sleeping, Swapped** → non può essere eseguito finché un qualche evento non si manifesta; il processo non è in memoria primaria
**Preempted** → il kernel ha appena tolto l’uso del processore a questo processo (*preemption*), per fare un context switch
**Created** → appena creato, ma non ancora pronto all’esecuzione
**Zombie** → terminato tutta la memoria del processo viene deallocata (compresa l’immagine) e l’unica cosa che sopravvive è il process control block con l’unico scopo di comunicare l’exit status al padre; una volta che il padre lo ha ricevuto, anche il PCB viene tolto e il processo figlio viene definitivamente terminato

### Processo Unix
Un processo in unix è diviso in:
- livello utente
- livello registro
- livello di sistema

#### Livello utente
**Process text** → il codice sorgente (in linguaggio macchina) del processo
**Process data** → sezione di dati del processo; compresi anche i valori delle variabili
**User stack** → stack delle chiamate del processo; in fondo contiene anche gli argomenti  con cui il processo è stato invocato
**Shared memory** → memoria condivisa con altri processi, usata per le comunicazioni tra processi

#### Livello registro
**Program counter** → indirizzo della prossima istruzione del process text da eseguire
**Process status register** → registro di stato del processore, relativo a  quando è stato swappato l’ultima volta
**Stack pointer** → puntatore alla cima dello user stack
**General purpose registers** → contenuto dei registri accessibili al programmatore, relativo a quando è stato swappato l’ultima volta

#### Livello sistema
**Process table entry** → puntatore alla tabella di tutti i processi, dove individua quello corrente
**U area** → informazioni per il controllo del processo
**Per process region table** → definisce il mapping tra indirizzi virtuali ed indirizzi fisici (page table)
**Kernel stack** → stack delle chiamate, separato da quello utente, usato per le funzioni da eseguire in modalità sistema

### Process Table Entry
![[Screenshot 2024-10-08 alle 00.39.39.png]]

### U-Area
![[Screenshot 2024-10-08 alle 00.40.22.png]]

### Creazione di un processo in Unix
La creazione di un processo unix tramite una chiamata di sistema `fork()`. In seguito a ciò, in Kernel Mode:
1. Alloca una entry nella tabella dei processi per il nuovo processo (figlio)
2. Assegna un PID unico al processo figlio
3. Copia l’immagine del padre, escludendo dalla copia la memoria condivisa (se presente)
4. Incrementa i contatori di ogni file aperto dal padre, per tenere conto del fatto che ora sono anche del figlio
5. Assegna al processo figlio lo stato Ready to Run
6. Fa ritornare alla fork il PID del figlio al padre, e 0 al figlio

Quindi, il kernel può scegliere tra:
- continuare ad eseguire il padre
- switchare al figlio
- switchare ad un altro processo

>[!info]
>Creare un processo a partire dal processo padre è il modo più efficiente di avviare un processo in quanto la maggior parte delle volte un programma inizia un processo a partire dal codice sorgente già esistente

---
## Thread
Finora abbiamo visto che ciascun processo compete con tutti gli altri alternando la loro esecuzione, tuttavia non è sempre così.Ci sono infatti delle applicazioni particolare (ad es. la maggior parte delle applicazioni GUI) che sono a loro volta organizzate in modo parallelo.
Infatti il programmatore dell’applicazione la ha suddivisa in diverse esecuzioni e ciascuna esecuzione è chiamata **thread**. Si tratta però di un processo, ma all’interno del processo è necessario di solito avere tre computazioni diverse (es. una monitora gli input, una che di conseguenza agli input ridisegni la finestra e una terza che faccia i calcoli richiesti) che devono poter avvenire contemporaneamente.

![[Pasted image 20241011134045.png]]

Diversi thread di uno stesso processo condividono le risorse **tranne** lo **stack** delle chiamate e il **processore**. Vengono quindi condivise le risorse input (un file aperto da un thread è disponibile anche a tutti gli altri thread), il codice sorgente ecc.

Teoricamente viene molto bene: si può dire che il concetto di processo incorpori le seguenti 2 caratteristiche
- gestione delle risorse (memoria, I/O ecc.)
- scheduling/esecuzione (stack e processore)
Dunque per quanto riguarda le risorse i processi vanno presi come un blocco unico, per quanto riguarda lo schedulung, i processi possono contenere diversi thread, e per questo vanno trattati in maniera diversa

![[Pasted image 20241011134132.png]]
Se ci troviamo in un sistema di processo a singolo thread troviamo la situazione affrontata fino ad ora (sinistra); se invece ci troviamo in un sistema a thread multiplo, non si ha solo l’immagine e il control block. Si ha infatti il:
- PCB e imagine comune (variabili globali ecc.) → parti condivise tra i thread
- Thread control block (uno per thread) → gestisce solo la parte di scheduling
- Stack dell’utente e stack di sistema (uno per thread)

Il vantaggio di usare diversi thread per un processo piuttosto che diversi processi sta nel fatto che con il thread diventa più sempre (e più efficiente) la creazione, la terminazione, fare lo switch tra thread è più efficiente di fare lo switch tra processi e la condivisione di risorse. Dunque ogni processo viene creato con un thread e il programmatore tramite opportune chiamate di sistema può:
- `spawn` → creare un nuovo thread (simile a `fork()` ma più leggera)
- `block` → per far bloccare un thread, non per I/O, ma esplicito (es. aspettare un altro thread)
- `unblock` → sbloccare un thread (es. un thread che ha finito di fare una computazione sblocca un altro thread che era in attesa)
- `finish` → terminare un thread

### ULT vs. KLT
I thread possono essere o a **livello utente** (User Level Thread) o a **livello di sistema** (Kernel Level  Thread)
![[Pasted image 20241011135159.png|center|440]]
Negli ULT, i thread non esistono a livello di sistema operativo (il SO genera il processo ed è totalmente ignaro dell’esistenza dei thread) e opportune librerie (esistenti solo a livello utente) si occupano di gestire i thread
Nei KLT, i thread esistono a livello di kernel e le librerie dei thread si appoggiano direttamente sulle system call del SO

**In base a cosa li scelgo?**
*Pro ULT*
Gli ULT sarebbero meglio in quanto per fare lo switch tra due thread dello stesso processo non è necessario fare il mode switch, le librerie necessarie sono infatti tutte contenute dentro la modalità utente. Permettono anche di avere una politica di scheduling diversa per ogni applicazione e di usare i thread sui SO che non li offrono nativamente

*Contro ULT*
Se un thread si blocca, si bloccano tutti i thread di quel processo (a meno che il blocco non sia dovuto alla chiamata `block` dei thread) in quanto il sistema operativo non sa nulla dei thread e quindi quando questo si blocca il SO blocca tutto il processo.
Se ci sono effettivamente più processori o più core, tutti i thread del processo ne possono usare solamente uno (si tratta sostanzialmente di alternarsi su un core di un processore)
Se il SO non ha i KLT, non possono essere usati i thread per routine del sistema operativo stesso

---
## Linux
### Processi e thread
Derivando da UNIX, che non ha i thread, la loro implementazione all’interno di Linux è stata particolarmente articolata ed è per questo che sono ben diversi da come sono stati mostrati fino ad ora.
In Linux l’unità di base sono i thread (è come se la `fork` creasse il thread), infatti i processi stessi sono chiamati Lightweight process (**LWP**).
In questo SO sono possibili **sia i KLT** (usati principalmente dal sistema operativo) **che gli ULT** (che possono essere direttamente scritti da un utente e che tramite la libreria `pthread` vengono essere poi mappati in KLT)

>[!warning]
>Il PID è un identificativo unico che vale per tutti i thread dello stesso processo viene quindi introdotto un **`tid`** (task identifier) che identifica ogni singolo thread. Come abbiamo detto ogni processo ha almeno un thread associato, questo thread ha il **TID uguale al PID**.
>Dunque è chiamato process identifier ma in realtà è un thread identifier, questo poiché l’unità di base è l’LWP, che coincide con il concetto di thread.
>L’entry del PCB che dà il PID comune a tutti i thread di un processo è il **`tgid`** (thread group identifier), e coincide con il PID del primo thread del processo
>Una chiamata a `getpid()` restituisce il `tgid`
>Ovviamente per processi a singolo thread `tgid` e `pid` coincidono

In Linux inoltre è presente **un PCB per ogni thread**, diversi thread dunque conterranno informazioni duplicate
![[Pasted image 20241011151649.png]]
`thread_info` → è organizzata per contenere anche il kernel stack, ovvero lo stack delle chiamate da usare quando il processo passa in modalità sistema (system call)
`thread_group` → punta agli altri thread dello stesso processo
`parent` e `real_parent` → puntano la padre del processo (ci sono anche link per i fratelli e figli)

### Stati dei processi
E’ sostanzialmente come quello a 5 stati (sono presenti i processi suspended ma non vi è fatta un’esplicita menzione)
Gli stati sono i seguenti:
- `TASK_RUNNING` → include sia Ready che Running (se uno processo è davvero running sta già sul processore non è dunque necessario distinguerli)
- `TASK_INTERRUPTIBLE`, `TASK_UNINTERRUPTIBLE`, `TASK_STOPPED`, `TASK_TRACED` → sono tutti Blocked che si differenziano per il motivo per cui sono blocked; in ordine blocked che non presenta problemi, connesso ad alcune operazioni I/O su dischi che sono particolarmente lenti per qualche motivo e non permette alcun tipo di azione, è stato esplicitamente bloccato, sto facendo debugging
- `EXIT_ZOMBIE`, `EXIT_DEAD` → entrambi stati di Exit

### Processi parenti
![[Pasted image 20241011153435.png|330]]
In ogni PCB sono presenti informazioni per risalire ai processi a lui connessi. Si può infatti accedere ai fratelli (processi che hanno lo stesso padre).
In aggiunta a questo ogni processo ha i suoi thread (non rappresentati)

### Segnali ed interrupt
Non bisogna confondere i segnali con interrupt (o eccezioni).
I **segnali** infatti possono essere inviati **da un processo utente ad un processo utente** (tramite una system call, chiamata `kill` ma potrebbe accadere che non termini il processo in rari casi).

Quando viene inviato un segnale, questo viene aggiunto all’opportuno campo del PCB del processo ricevente. A questo punto, quando il processo viene nuovamente schedulato per l’esecuzione, il kernel controlla prima se ci sono segnali pendenti; se si esegue un’opportuna funzione chiamata *signal handler* (a differenza dell’interrupt handler, questo viene eseguito in **user mode**). I signal handler possono essere di sistema possono essere sovrascritti da signal handler definiti dal programmatore (alcuni segnali hanno handler non sovrascribili)

I segnali possono anche essere inviati da un processo in modalità sistema, ma in questo caso molto spesso ciò è dovuto ad un interrupt a monte.
Esempio tipico: eseguo un programma C scritto male, che accede ad una zona di memoria senza averla prima richiesta, il processore fa scattare un’eccezione, viene eseguito l’opportuno exception handler (in kernel mode) che essenzialmente manda il segnale `SIGSEGV` (violazione di segmento, segmentation fault) al processo colpevole, quando il processo colpevole verrà selezionato nuovamente per andare in esecuzione, il kernel vedrà dal PCB che c’è un segnale pendente, e farà in modo che venga eseguita l’azione corrispondente. L’azione di default per tale segnale è di far terminare il processo (fa una system call che ritorna in kernel mode e termina il processo; può essere riscritta dall’utente quando verrà eseguita tale azione, sarà in user mode)

Dunque le differenze fondamentali tra segnali ed interrupt sono:
- i signal handler sono eseguiti in user mode mentre gli interrupt handler in kernel mode
- i signal handler potrebbero essere riscritto dal programmatore mentre gli interrupt handler no
