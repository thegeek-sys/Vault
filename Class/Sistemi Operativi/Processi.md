---
Created: 2024-10-03
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Requisiti di un SO
Il compito fondamentale di un sistema operativo è quello di **gestire i processi**, deve quindi gestire tutte le varie computazioni.
Un sistema operativo moderno deve quindi:
- permettere l’esecuzione alternata di processi multipli → anche se ho meno processori che processi questo non deve essere un problema, bisogna far alternare i processi in esecuzione sui processori disponibili
- assegnare le risorse ai processi (es. un processo richiede l’uso della stampante)
- permettere ai processi di scambiarsi informazioni
- permettere la sincronizzazione tra processi

---
## Cos’è un processo?
Un **processo** è un’istanza di un programma, memorizzato generalmente sul archiviazione di massa, in esecuzione (ogni singola istanza di fatti dà vita a un processo).
Potrebbe però anche essere visto come un’entità che può essere assegnata ad un processore in esecuzione tipicamente caratterizzata dall’esecuzione da una sequenza di istruzioni di cui voglio conoscere l’esito, uno stato corrente e da un insieme associato di risorse

Questo è quindi composto da:
- **codice** → ovvero le istruzioni da eseguire
- un insieme di **dati**
- un numero di **attributi** che descrivono il suo stato

Un processo ha 3 macrofasi: creazione, esecuzione, terminazione. Quest’ultima può essere prevista (quando il programma è terminato o quando l’utente chiude volontariamente in programma tramite la X) oppure non prevista (processo esegue un’operazione non consentita che potrebbe risultare con una terminazione involontaria del processo)

---
## Elementi di un processo
Finché un processo è in esecuzione ad esso sono associati un certo insieme di informazioni, tra cui:
- identificatore
- stato (running etc.)
- priorità
- hardware context → attuale situazione dei registri
- puntatori alla memoria principale (definisce l’immagine del processo)
- informazioni sullo stato dell’input/output
- informazioni di accounting (quale utente ha eseguito il processo)

### Process Control Block
Per ciascuno processo attualmente in esecuzione è presente un **process control block**. Si tratta di un insieme di informazioni (gli elementi di un processo) raccolte insieme e mantenute nella zona di memoria riservata al kernel.
Questo viene interamente creato e gestito dal sistema operativo e il suo scopo principale è quello di permettere al SO di **gestire più processi contemporaneamente** (contiene  infatti le  informazioni sufficienti per bloccare un programma e farlo riprendere più tardi dallo stesso punto in cui si trovava)

---
## Traccia di un processo
Un ulteriore aspetto importante in un processo è la **trace** ovvero l’insieme di istruzioni di cui è costituito un processo. Il **dispatcher** invece è un piccolo programma che sospende un processo per farne andare un altro in esecuzione

### Esempio
Si considerino 3 processi in esecuzioni, tutti caricati in memoria principale
![[Screenshot 2024-10-04 alle 10.57.44.png|140]]

La traccia, dal punto di vista del processo, appare come l’esecuzione sequenziale delle istruzioni del singolo processo
![[Screenshot 2024-10-04 alle 11.00.20.png]]

La traccia, del punto di vista del processore, ci mostra come effettivamente vengono eseguiti i 3 processi
![[Screenshot 2024-10-04 alle 11.02.08.png]]
>[!note] Le righe in blu sono gli indirizzi del dispatcher

---
## Modello dei processi a 2 stati
Un processo potrebbe essere in uno di questi due stati (non consideriamo infatti creazione e terminazione)
- in esecuzione
- non in esecuzione (anche quando viene messo in pausa dal dispatcher)
![[Screenshot 2024-10-04 alle 11.05.44.png|center|450]]

Dal punto di vista dell’implementazione avremmo una coda in cui sono processi tutti i processi che non sono in esecuzione, il dispatch quindi prende il processo in cima alla queue (quello in nero) e lo mette in esecuzione.
I processi vengono quindi mossi dal dispacher dalla CPU alla coda e viceversa, finché il processo non viene completato
![[Screenshot 2024-10-04 alle 11.09.52.png|400]]

---
## Creazione e terminazione di processi
### Creazione
In ogni istante in un sistema operativo sono $n\geq 1$ processi in esecuzione (come minimo ci sta un’interfaccia grafica o testuale in attesa di input). Quando viene dato un comando dall’utente, quasi sempre si crea un nuovo processo

**process spawning** → è un fenomeno che si verifica quando un processo in esecuzione crea un nuovo processo. Si hanno quindi un **processo padre** (quello che crea) e un **processo figlio** (processo creato) passando quindi da $n$ a $n+1$ processi

### Terminazione
Con la terminazione si passa da $n\geq 2$ processi a $n-1$. Esiste in oltre sempre un processo “*master*” che non può essere terminato (salvo spegnimento del computer)
#### Normale completamento
Il normale completamento di un processo (come precedentemente nominato) avviene con l’istruzione macchina `HALT` che genera un’interruzione (che nei linguaggi di alto livello è invocata da una system call, inserita automaticamente dai compilatori dopo l’ultima istruzione di un programma)
#### Uccisioni
Oltre al normale completamento ci stanno le uccisioni, eseguite generalmente dal SO a causa di errori come ad esempio:
- memoria non disponibile
- errori di protezione
- errore fatale a livello di istruzione (divisione per zero ecc.)
- operazione di I/O fallita
Oppure dall’utente (es. X sulla finestra) o da un altro processo (es. invio segnale da terminale)

---
## Modello dei processi a 5 stati
In questo caso il momento in cui il processo non è in esecuzione è diviso tra quando il processo è ready e quando il processo è blocked
![[Screenshot 2024-10-04 alle 12.14.57.png|center|500]]
Una volta creato un processo diventa subito ready, può diventare running attraverso il dispatch. Se sono running e ad un certo punto nelle istruzioni eseguo un’operazione I/O entro in attesa (blocked) e ci rimango finché il dispositivo I/O non ha terminato e a quel punto ritorno ready

![[Screenshot 2024-10-04 alle 12.19.05.png|450]]
Avendo due stati necessiterà quindi di due code (al posto di una sola). Si aggiunge  infatti la coda di blocked.

Assumiamo che tutti questi processi siano in RAM, ci sono svariati motivi per cui un processo possa esser messo in attesa, infatti i sistemi operativi non hanno una coda per tutti gli eventi, ma ne hanno una per ogni evento (per ogni motivo per cui il processo è stato messo in attesa)
![[Screenshot 2024-10-04 alle 12.24.11.png|440]]

---
## Processi sospesi
Potrebbe succedere che molti processi possano essere in attesa di input/output. Finché sono blocked questi stanno solamente occupando inutilmente della memoria RAM. Dunque quello che viene fatto è spostare (*swappare*) alcuni processi sul disco e, quando dovranno essere ripresi, vengono rispostati sulla RAM.
Dunque lo stato “blocked” diventa “suspendend” quando viene swappato su disco

Questo porta all’introduzione di due nuovi stati:
- *blocked/suspended* → swappato mentre era bloccato
- *ready/suspended* → swappato mentre non era bloccato

Abbiamo quindi un totale di 7 stati:
![[Screenshot 2024-10-04 alle 12.30.12.png|center|500]]
Adesso quindi una volta creato il processo il SO decide se metterlo in RAM (Ready) oppure swapparlo su disco (Ready/Suspend). Il dispatch sceglie solo tra i processi in RAM. Quando il processo fa qualche richiesta bloccante, diventa blocked, da cui si sblocca solo quando la richiesta soddisfatta. Un processo blocked può essere swappato su disco. Da Running anche può essere swappato

### Motivi per sospendere un processo

| Motivo                       | Commento                                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Swapping                     | Il SO ha bisogno di eseguire un processo con alta priorità (o è molto grande) e per fargli spazio bisogna swappare dei processi ready |
| Interno al SO                | Il SO sospetta che il processo stia causando problemi                                                                                 |
| Richiesta utente interattiva | Ad esempio: debugging                                                                                                                 |
| Periodicità                  | Il processo viene eseguito periodicamente e può venire sospeso in attesa della prossima esecuzione                                    |
| Richiesta del padre          | Il padre potrebbe voler sospendere l’esecuzione di un figlio per esaminarlo o modificalo per coordinare l’attività tra più figli      |

---
## Processi e Risorse
Il sistema operativo oltre a dover gestire i processi deve anche gestire come questi richiedono e acquisiscano delle risorse (tipicamente dispositivi di input).

![[Screenshot 2024-10-04 alle 12.42.03.png|520]]
Con le linee piene si intendono risorse che sono state acquisite dai processi, mentre con le linee tratteggiate delle risorse che sono state solamente richieste dai processi (ma non concesse, ad esempio perché è una risorsa esclusiva, ovvero fin tanto che lo ha $P_{1}$ non lo può avere $P_{2}$)

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

---
## Modalità di esecuzione
La maggior parte dei processori supporta almeno due modalità di esecuzione (il Pentinum ne ha 4):
- modo sistema (*kernel mode*) → in cui si ha il pieno controllo e si può accedere a qualsiasi locazione di memoria (compresa quella del kernel)
- modo utente → molte operazioni sono vietate

### Kernel Mode
La **kernel mode** serve per le operazioni eseguite dal kernel. Viene utilizzata per:
- gestione dei processi (tramite PCB)
	- creazione e terminazione
	- pianificazione di lungo, medio e breve termine (*scheduling* e *dispatching*)
	- avvicendamento (*process switching*)
	- sincronizzazione e comunicazione
- gestione della memoria principale
	- allocazione di spazio per i processi
	- gestione della memoria virtuale
- gestione dell’I/O
	- gestione dei buffer e delle cache per l’I/O
	- assegnazione risorse I/O ai processi
- funzioni di supporto (gestione interrupt, accounting, monitoraggio)

### Da User Mode a Kernel Mode e ritorno
Si basa su un’idea semplice: un processo utente inizia sempre in modalità utente, ma cambia e si porta in modalità sistema in seguito ad un interrupt (come abbiamo visto infatti una volta eseguita un’interrupt viene eseguita una parte di codice di sistema).
La prima cosa che fa l’hardware, prima di cominciare alla procedura di sistema da eseguire, cambia modalità passando in kernel mode; questo permette di eseguire l’interrupt handler in kernel mode.
L’ultima istruzione dell’interrupt handler, prima di ritornare al processo di partenza, ripristina la modalità utente

Dunque un processo utente può cambiare modalità a sé stesso, ma **solo per eseguire software di sistema**.
Si ha quindi questo cambiamento in seguito a (esplicitamente voluti):
- system call
- in risposta ad una sua precedente richiesta di I/O (in generale di risorse)
Codice eseguito per conto dello stesso processo interrotto, che non lo ha esplicitamente voluto:
- errore fatale (*abort*) → il processo spesso viene terminato
- errore non fatale (*fault*) → viene eseguito un qualcosa prima di tornare in user mode e continuare il processo
Codice eseguito per conto di qualche processo. In particolare avviene quando un processo A ha fatto una richiesta I/O quindi il SO lo ho messo in blocked, il SO intanto mette in esecuzione un secondo processo B, nel mentre viene esaudita la richiesta di A ma ciò avviene per conto del processo B

### System call sui Pentium
Il codice per una system call sui Pentium è strutturata così:
1. si preparano gli argomenti della chiamata mettendoli in opportuni registri
	tra di essi ci sta il numero che identifica la system call
2. esegue l’istruzione `int 0x80`, che appunto solleva un interrupt (in realtà un’eccezione)
2. in alternativa, dal Pentium 2 in poi, può eseguire l’istruzione `sysenter`, che omette alcuni controlli inutili
Da notare che anche creare un nuovo processo è una system call: in Linux *fork* (oppure *clone* più generale)

---
## Creazione di un processo
Per creare un processo il sistema operativo deve:
1. Assegnargli un PID unico
2. Allocargli spazio in memoria principale
3. Inizializzare il process control block
4. Inserire il processo nella giusta coda (es. ready oppure ready/suspended)
5. Creare o espandere altre strutture dati (es. quelle per l’accounting)

---
## Switching tra processi
Lo **switching tra processi** consiste nel concedere il processore ad un altro processo per qualche motivo scaturito da un evento ed è l’operazione più delicata quando si tratta di scrivere un sistema operativo in quanto pone svariati problemi tra cui:
- Quali eventi determinano uno switch?
- Cosa deve fare il SO per tenere aggiornate le strutture dati in seguito ad uno switch tra processi?

> [!warning] Attenzione a distinguere lo switch di modalità (da utente a sistema) e lo switching di processi

### Quando effettuare uno switch?
Uno switch può avvenire per i seguenti motivi

| Meccanismo     | Causa                                             | Uso                                                                                                                                     |
| -------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Interruzione   | Esterna all’esecuzione dell’istruzione corrente   | Reazione ad un evento esterno asincrono; include i quanti di tempo per lo scheduler (per evitare che il processore venga monopolizzato) |
| Eccezione      | Associata all’esecuzione dell’istruzione corrente | Gestione di un errore sincrono                                                                                                          |
| Chiamata al SO | Richiesta esplicita                               | Chiamata a funzione di sistema (caso particolare di eccezione)                                                                          |

### Passaggi
Quando si deve sostituire un processo per prima cosa si switcha in kernel mode poi:
1. Si salva il contesto del programma (registri e PC salvati nel PCB di quel processo)
2. Aggiornare il process control block per quanto riguarda lo stato, attualmente in running
3. Spostare il process control block nella coda appropriata: ready, blocked, ready/suspended
4. Scegliere un altro processo da eseguire
5. Aggiornare lo stato del process control block del processo selezionato
6. Aggiornare le strutture dati per la gestione della memoria
7. Ripristinare il contesto del processo selezionato
