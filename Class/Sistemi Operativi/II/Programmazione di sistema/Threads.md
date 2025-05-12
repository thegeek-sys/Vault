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

### Modello “da molti a 1”
