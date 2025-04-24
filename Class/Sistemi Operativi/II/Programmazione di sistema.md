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

### Gestione errori system call
L’esecuzione di una system call può interrompere e non andare a buon fine per diversi motivi, principalmente per:
- il processo che la invoca non ha sufficienti privilegi per l’esecuzione
- non ci sono sufficienti risorse per l’esecuzione
- gli argomenti in ingresso alla system call non sono validi

Per questi motivi è fondamentale controllare i valori di ritorno per rilevare e segnalare all’utente il verificarsi di errori. Per un corretto funzionamento del programma è inoltre **fondamentale gestire in maniere opportuna** l’eventuale errore verificatosi

#### $\verb|errno|$
La variabile globale `errno` rappresenta il codice di errore dell’ultima system call invocata che ha generato un errore (le syscall che terminano con successo lasciano `errno` invariato)

Di fatto le system call che terminano con un errore tipicamente ritornano il valore $-1$ e impostano `errno` con il codice specifico dell’errore che si è generato durante l’esecuzione

#### $\verb|perror()|$
Questa è una funzione della libreria standard
```c
#include <stdio.h>
void perror(const char *prefix);
```

`perror` stampa su `stderr` il messaggio di errore, convertendo il codice di errore `errno` in una stringa formata da
```
<prefix>:<errno_string>
```
dove `errno_string` rappresenta il messaggio di errore in formato di stringa (e quindi mnemonico) associato al valore di `errno` (es. `perror("main");` invia su srderr la stringa `main:mess_errore_mnemonico=errno`)

#### $\verb|strerror()|$
Questa è una funzione di libreria che consente di convertire un codice di errore numerico `errno` (che acquisisce come parametro di input nella sua equivalente rappresentazione in stringa)
```c
#include <string.h>
char *strerror(int errnum);

// ad esempio
printf("Si è verificato errore:%s\n", strerror(errno));
```

#### Esempio di gestione errore
```c
..
#include <syscall_lib.h>
#include <stdio.h> /* per perror() */
#include <string.h> /* per strerror() */
#include <errno.h>
..
if (una-syscall() == -1) {
	int errsv = errno;
	perror("main"); // stderr
	printf(”Si è verificato errore:%s\n”, strerror(errsv)); //stdout
	
	if (errsv == ...) {
		...
	}
} ..
```

### Debug syscall
E’ spesso utile monitorare il comportamento di un processo relativamente all’invocazione di system call. In nostro aiuto ci sta il comando `strace` che permette di tracciare l’invocazione di system call da parte di un processo. In particolare consente di stampare la lista di system call invocate da un processo e relativi parametri

```bash
strace -o strace.txt -s <max_string_size> /path2/program
strace -p <pid> -e trace=syscall1,syscall2... -o strace_<pid>.txt
strace -o passwd.strace -s 100 cat /etc/passwd
```

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

---
## Allocazione di memoria

```c
#include <stdlib.h> // funzioni di libreria, non syscall

// allocano nell'heap
void *malloc(size_t size);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);

#include <alloca.h> // alloca nello stack
void *alloca(size_t size);
```

### $\verb|mmap|$, $\verb|brk|$, $\verb|sbrk|$ system call
Le `m/c/ralloc` usano le vere system call per la gestione della memoria (es. `mmap` alloca memoria, `brk` cambia la dimensione data segment di un processo)

```c
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
```