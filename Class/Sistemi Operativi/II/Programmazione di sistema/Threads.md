---
Created: 2025-05-12
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## Applicazioni multithread
In una applicazione tradizionale, il programmatore definisce un unico flusso di esecuzione delle istruzioni. La CPU esegue istruzioni macchina in sequenza e il flusso di esecuzione “segue” la logica del programma (cicli, funzioni, chiamate di sistema, gestori di segnali…)
Quando il flusso di esecuzione arriva ad eseguire la API `exit()` l’applicazione termina

Le applicazioni multithread consentono al programmatore di definire diversi flusso di esecuzione:
- ciascun flusso di esecuzione condivide le strutture dati principali dell’applicazione
- ciascun flusso di esecuzione procede in modo concorrente ed indipendente dagli altri flussi
- l’applicazione finisce solo quando tutti i flussi di esecuzione vengono terminati

>[!hint]
>Ciascun thread compie il proprio lavoro eseguendo un flusso di istruzioni indipendente e cooperando con gli altri thread

>[!example] Esempio di applicazione multithread
>Un browser Web potrebbe essere costituito dai seguenti thread
>- thread principale di controllo dell’applicazione
>- thread per l’interazione con l’utente
>- thread per la visualizzazione (rendering) delle pagine in formato HTML
>- thread per la gestione dei trasferimenti di pagine e file dalla rete
>- thread per l’esecuzione dei frammenti di script integrati nelle pagine Web
>- thread per l’esecuzione dei programmi Java, Flash, ecc.

### Motivazioni
Il motivo principale dell’utilizzo dei threads è l’**elevato parallelismo interno dei calcolatori elettronici**:
- *DMA* → trasferimento dati tra macchina primaria e periferiche di I/O senza intervento della CPU
- *hyperthreading* → supporto a diversi flussi di esecuzione, ciascuno con un proprio insieme di registri, che si alternano sulle unità funzionali della CPU
- *multicore* → diversi core di calcolo integrati sullo stesso chip e che condividono alcune risorse hardware quali cache di 2° livello, MMU, …
- *multiprocessori* → diverse CPU integrate sulla stessa scheda madre

>[!info]
>E’ difficile scrivere applicazioni tradizionali (unico flusso) che sfruttino a fondo il parallelismo interno al calcolatore

### Vantaggi
- **Riduzione del tempo di risposta**
	- anche se una parte dell’applicazione è bloccata in attesa di eventi esterni, un altro thread può essere eseguito per interagire con l’utente o gestire altri eventi
- **Migliore condivisione delle risorse**
	- tutti i thread di una applicazione condividono le risorse (strutture di dati in memoria, file aperti), e la comunicazione tra i thread è immediata
- **Maggiore efficienza**
	- rispetto ad una applicazione costituita da più processi cooperanti, l’applicazione multithread è più efficiente, perché il SO gestisce i thread più rapidamente. In Linux, creare un thread richiede 1/10 del tempo richiesto per la creazione di un processo
- **Maggiore scalabilità**
	- i thread possono sfruttare in modo implicito il parallelismo interno del calcolatore

---
## Processi e threads
![[Pasted image 20250512225949.png]]

Un processo per una applicazione monothread è costituito da:
- codice → istruzioni macchina in memoria
- strutture dati → variabili globali in memoria, heap
- file aperti
- contenuto dei registri della CPU → contesto
- posizione e contenuto dello stack UM

In una applicazione multithread alcune risorse sono *comuni* e condivise tra tutti i thread come **codice**, **strutture dati** e **file aperti**
Altre risorse invece sono *private* per ciascun thread come il **contenuto dei registri della CPU** (contesto) e la **posizione e contenuto dello stack UM**

| Processi                                                                                       | Thread                                                                                                                                   |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| istanze di un programma in esecuzione (heavyweight)                                            | componente di un processo e più piccola unità di esecuzione (lightweight)                                                                |
| cambio di contesto (context switching) richiede interazione con SO                             | context switch non richiede interazione con SO                                                                                           |
| ogni processo ha il suo spazio di memoria                                                      | usano la memoria del processo a cui appartengono                                                                                         |
| richiedono più risorse di sistema                                                              | richiedono meno risorse di sistema                                                                                                       |
| difficili da creare                                                                            | facili da creare                                                                                                                         |
| comunicazione tra processi lenta in quanto ogni processo ha un differente indirizzo di memoria | comunicazione tra thread è veloce in quanto i thread condividono lo stesso indirizzo (e area) di memoria del processo a cui appartengono |
| ogni processo eseguire indipendentemente (isolato)                                             | ogni thread può leggere, scrivere, modificare dati di altri thread                                                                       |

---
## Implementazione di applicazioni multithread
Esistono principalmente due componenti in un’implementazione delle applicazioni multithread:
- implementazione a livello utente
- implementazione a livello kernel

In sostanza tutte le implementazioni sono caratterizzate dalla relazione tra:
- thread utente → il flusso di esecuzione dell’applicazione
- thread kernel → l’astrazione (strutture dati, servizi) definita all’interno del nucleo dell’SO per gestire un flusso di esecuzione

>[!warning]
>La definizione di thread utente o thread kernel non è basata su una diversa modalità d’esecuzione (User Mode o Kernel Mode)

### User-level vs Kernel-level threads

| User level                                                         | Kernel level                                                                        |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| SO inconsapevole dei threads                                       | SO consapevole dei threads                                                          |
| Facile implementazione dei thread                                  | Difficile implementazione dei threads                                               |
| Context switch veloce, senza supporto hardware                     | Context switch e costoso a livello di tempo e risorse, e richiede supporto hardware |
| Una chiamata bloccante fatta da un thread blocca l’intero processo | Una chiamata bloccante fatta da un thread non blocca l’intero processo              |
| ss. POSIX, Java threads                                            | es. Windows, Solaris                                                                |

### Modello “da molti a 1”
Una applicazione multithread è costituita da un singolo processo del SO e a diversi thread utente corrisponde un singolo thread kernel (un singolo processo tradizionale)

Il nucleo del SO non è coinvolto nella gestione dei flussi dell’applicazione, la quale (eventualmente usando una libreria di sistema) gestisce autonomamente i thread utente. In particolare schedula i vari flussi di esecuzione e salva e ripristina gli stack UM e i contesti

Se un thread invoca una chiamata di sistema bloccante, il processo (e quindi tutti i thread utente) vengono bloccati. Risulta quindi impossibile sfruttare in modo implicito il parallelismo interno del calcolatore (implementazione detta “a livello utente” o green threads)

### Modello “da 1 a 1”
In questo modello ciascun thread dell’applicazione corrisponde ad un singolo thread kernel

Per cui il nucleo del SO si occupa della gestione e schedulazione dei thread kernel (perciò gestisce anche i thread utente) e l’applicazione utilizza le API definite in una libreria di sistema per creare e distruggere i thread utente e per gestire comunicazione e sincronizzazione
La libreria implementa i servizi richiesti dall’applicazione invocando opportune chiamate di sistema

Ciascun thread utente può invocare chiamate di sistema bloccanti senza bloccare gli altri thread. Dunque l’applicazione sfrutta in modo implicito il parallelismo interno del calcolatore (implementazione detta “a livello kernel“, o native threads)

### Modello “da molti a molti”
Gli $n_{u}$ thread utente della applicazione corrispondono a $n_{k}$ thread kernel (con $n_{k}\leq n_{u}$)

Il nucleo del SO si occupa della gestione e schdulazione dei thread kernel (non gestisce direttamente i thread utente) e l’applicazione utilizza le API definite in una libreria di sistema che definisce il numero $n_{k}$ di thread kernel, crea e distrugge i thread utente e mappa i thread utente sui thread kernel
La libreria di sistema:
- usa chiamate di sistema per gestire i thread kernel
- gestisce la schedulazione e lo stack UM dei thread utente mappati sullo stesso thread kernel
- gestisce comunicazione e sincronizzazione

### Esempi di implementazione
Esempi di implementazioni con modello “da molti a uno” (possono essere utilizzate con qualunque sistema operative):
- la libreria green threads disponibile in Sun Solaris
- la libreria GNU Pth (GNU Portable Threads)

Tutti i maggiori SO oggi supportano i thread nativi, e dunque i modelli “da uno a uno” e “da molti a molti”:
- Linux, MS Windows, MacOS X tendono ad adottare il modello “da uno a uno”
- IRIX, HP-UX, Tru64 UNIX adottano il modello “da molti a molti”

In ogni caso la scelta del modello “da uno a uno” piuttosto che “da molti a molti” dipende dalla libreria di thread utilizzata, e non dal sistema operativo

### Librerie dei thread
Generalmente il programmatore utilizza una libreria di sistema per realizzare una applicazione multithread che offre API non direttamente correlate con la tipologia di thread utilizzata
Possono infatti esistere diverse versioni di una libreria con identiche API ma thread a livello utente oppure kernel (es. `pthreads`, POSIX threads)

Alcune librerie e le relative API sono invece specifiche per un determinato sistema operativo e tipologia di thread (es. libreria per i thread delle API Win32)
Altre librerie, e le relative API, invece sono specifiche di un linguaggio ad alto livello rendendo l’uso della libreria implicito e automatico (la libreria utilizza una libreria di thread di livello più basso), ne è un esempio la libreria di thread del linguaggio Java

#### $\verb|pthreads|$
La libreria `pthreads` è definita dallo standard POSIX (IEEE 1003.1c) che definisce le API, ma non stabilisce quale debba essere la loro implementazione in uno specifico SO

In Linux sono coesistite tre diverse implementazioni:
- LinuxThreads → prima implementazione basata sul modello “da uno a uno”, non più supportata
- NGPT (Next Generation POSIX Threads) → sviluppata da IBM sul modello “da molti a molti”, non più supportata
- NPTL (Native POSIX Threads Library) → ultima implementazione, più efficiente e più aderente allo standard, basata sullo standard “da uno a uno”

Oggi in Linux si utilizza esclusivamente NPTL

---
## Implementazione
### Creazione di un nuovo thread in $\verb|pthreads|$

```c
int pthread_create(ptid, pattr, start, arg)
```

La funzione di libreria `pthread_create()` crea un nuovo thread. Analizziamo gli argomenti:
- `ptid` → puntatore a variabile di tipo `pthread_t` che conterrà l’identificatore del nuovo thread (TID)
- `pattr` → puntatore ad una variabile contenente attributi (flag) per la creazione del thread (opzionale)
- `start` → funzione inizialmente eseguita dal thread, con recipe `void *start(void *)`
- `arg` → puntatore passato come argomento a `start()`