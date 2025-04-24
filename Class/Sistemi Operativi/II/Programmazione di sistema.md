---
Created: 2025-04-24
Class: "[[Sistemi Operativi]]"
Related:
---
---
## Introduction
Il **kernel** è la componente del sistema operativo che gestisce le risorse disponibili ed offre l’accesso e l’utilizzo delle risorse da parte dei processi
Le risorse principali gestite sono:
- CPU
- RAM
- I/O

Le **system call** permettono ad un processo di accedere ai servizi offerti dal kernel, permettendo quindi al programmatore di creare programmi che interagiscono direttamente con il kernel

---
## System call
![[Pasted image 20250424094111.png|center]]
Per ogni system call esiste un omonimo comando in C. Infatti quando il processo utente invoca questa funzione usando la sequenza standard C, questa funzione invoca il corrispondente servizio del kernel utilizzando la tecnica necessaria (ad esempio mettendo i parametri della funzione C in un registro e poi generando un interrupt per il kernel usando istruzioni macchina)

Le system call sono utilizzate per:
- file
	- file e directory → creazione, accesso/modifica contenuto/attributi
	- IPC (*Inter Process Communication*) → pipe e fifo (processi residenti sulla stessa macchina)
	- socket e socket di rete (networking) → comunicazione tra processi mediante protocolli di rete
- gestione della memoria
- processi
	- gestione di processi → creazione/terminazione, esecuzione, sincronizzazione, accesso/modifica attributi/ambiente
	- gestione dei thread (POSIX Thread) → processi leggeri, creazione/terminazione, sincronizzazione
	- segnali → interazione/comunicazione tra processi, creazione sezioni critiche

### Dove trovare la descrizione delle syscall
Nella sezione 2 del `man` sono contenute informazioni dettagliate sull’utilizzo ed il funzionamento delle system call

```bash
man 2 nome_system_call
```

### Gestione errory system call
L’esecuzione di una system call può interrompere 

---
## Funzioni di libreria general purpose
A differenza delle system call, le funzioni di libreria general purpose non sono punti di accesso ai servizi del kernel, ma possono invocare una o più system call (ad esempio `printf` può usare la system call `write` per scrivere una stringa in output), ma anche nessuna (es. `strcpy`, `atoi`)

### Dove trovare la descrizione delle funzioni general purpose
Nella sezione 2 del `man` sono contenute informazioni dettagliate sull’utilizzo ed il funzionamento delle funzioni di libreria

```bash
man 2 nome_funz_libreria
```

### Differenza tra system call e funzioni general purpose
#### Analogie
- entrambe sono funzioni C
- entrambe forniscono funzioni ad un’applicazione
#### Differenze
- una funzione general purpose può essere rimpiazzata ma una system call no → la syscall `skbr` permette di allocare memoria e la funzione `malloc` è implementata usando la `skbr`; dunque possiamo implementare la `malloc` ma dovremo sempre utilizzare la `sbrk`
- le system call introducono una sperazione di compiti → la `sbrk` alloca chunk di memoria (kernel mode) per il processo utente, mentre la `malloc` gestisce l’area di memoria in user mode
- le funzioni di libreria semplificano l’uso delle system call → le system call espongono un’iterfaccia minimale, mentre le funzioni di libreria forniscono funzionalità elaborate e semplificano la gestione delle strutture dati di input e output (es. la syscall `time` restituisce lo Unix time mentre `ctime` restituisce la data attuale)
