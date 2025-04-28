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

### $\verb|brk|$, $\verb|sbrk|$ system call
Le `m/c/ralloc` usano le vere system call per la gestione della memoria (es. `mmap` alloca memoria, `brk` cambia la dimensione data segment di un processo)

> [!example]
> ```c
> char *strPtr=NULL;
> const int SIZE_OF_ARRAY=30;
> strPtr=(char *) calloc(SIZE_OF_ARRAY, sizeof(char) );
> ```

```c
int brk(void *addr)
```
La funzione `brk()` viene utilizzata per modificare lo spazio assegnato per il processo chiamante. La modifica viene apportata impostando il valore di interruzione del processo su `addr` e allocando la quantità di spazio appropriata. La quantità di spazio allocato aumenta con l'aumento del valore di interruzione. Lo spazio appena assegnato è impostato su 0. Tuttavia, se l'applicazione prima diminuisce e poi aumenta il valore di interruzione, il contenuto dello spazio riassegnato non viene azzerato.

### $\verb|realloc|$

```c
void *realloc(void *ptr, size_t size)
```
Questa funzione permette di modificare la dimensione dell’area di memoria precedentemente allocata con `m/calloc` e puntata da `ptr` nella dimensione specificata dal valore di `size`. Ritorna `NULL` in caso di errore (ma l’area di memoria originale rimane intatta)

> [!example]
> ```c
> char *strPtr=NULL;
> const int SIZE_OF_ARRAY=30;
> strPtr=(char *) calloc(SIZE_OF_ARRAY, sizeof(char));
> strPtr1=(char *) realloc(strPtr, 10*SIZE_OF_ARRAY);
> ```

>[!hint] `strptr1` potrebbe essere diverso da `strptr`
>Infatti, nel caso di aumento della dimensione, qualora non riuscisse ad allargare l’area correttamente allocata e puntata da `ptr`, allora una nuova area liberando quella correttamente puntata da `ptr` e copiando tutti i contenuti all’interno di `ptr`

E’ inoltre importante ricordare che la nuova area di memoria allocata non viene inizializzata (se ad esempio quella originale lo era)

---
## Memory leakage
L’esecuzione di un programma che non gestisce correttamente la liberazione della memoria non più utilizzata, può causare un aumento del consumo della memoria del sistema.
Questo può portare al fallimento del programma stesso, non riuscendo più ad allocare altra memoria da utilizzare, ed in generale, può portare al deterioramento delle performance e del funzionamento del sistema.

Dunque è importante usare sempre `free()` dopo aver terminato l’uso della memoria allocata dinamicamente

### $\verb|memset()|$ e $\verb|memcpy()|$
```c
void *memset(void *s, int c, size_t n);
```
La funzione `memset()` assegna il valore intero `c` ad `n` bytes contigui dell’area di memoria puntata da `s`

```c
void *memcpy(void *dest, const void *src, size_t n);
```
Copia `n` bytes contigui a partire da `src` in `dest`, però le due aree di memoria non devono sovrapporsi (può risultare ad esempio utile per duplicare rapidamente un array)

### $\verb|mmap|$
```c
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
```

Questa syscall crea un area di memoria per mappare un file a partire da un indirizzo
specificato, con livello di protezione indicato (lettura, scrittura, esecuzione) e, come altre funzioni di gestione della memoria in C, ritorna un puntatore (serve fare il casting al tipo di puntatore relativo al tipo di dato contenuto nella memoria per poter utilizzare correttamente l’aritmetica dei puntatori)

- `addr` → indirizzo iniziale dell’area di memoria in cui vogliamo mappare il file (se `addr=0` sceglie il sistema)
- `fd` → file descriptor (il file va aperto prima)
- `len` → il numero di byte da trasferire
- `off` → offset nel file
- `prot` → indica il livello di protezione
	- `PROT_READ` → solo lettura
	- `PROT_WRITE` → solo scrittura
	- `PROT_EXEC` → sola esecuzione
	- `PROT_NONE` → nessun accesso
- `flags`
	- `MAP_SHARED` → le operazioni modificano il file (R/W)
	- `MAP_PRIVATE` → viene creata una copia private del file mappato e solo questa viene modificata

Si preferisce usare `mmap` a `malloc` se:
- serve un controllo preciso su protezioni e indirizzamento
- si vuole mappare file in memoria (es. modificare un file grande senza copiarlo)
- si vuole memoria condivisa fra processi diversi
- si gestiscono grossi blocchi di memoria (`mmap` è più efficiente per allocazioni molto grandi)

![[Pasted image 20250428123232.png|center|450]]

E’ dunque possibile mappare un file su disco in un area di memoria (`buffer`), per cui la lettura/scrittura dal/sul `buffer` risultano in lettura/scrittura dal/sul disco (senza accedere al disco)

---
## Sync e demapping
```c
int msync(void *addr, size_t len, int flags);
```

`msync` è una funzione che serve a sincronizzare una regione di memoria mappata (ottenuta con `mmap`) con il file o il dispositivo da cui quella memoria è stata creata. Senza `msync` le modifiche potrebbero rimanere solo in memoria (RAM) e non finire mai fisicamente sul file

>[!warning]
>E’ possibile usarlo solo se la memoria è `MAP_SHARED`

```c
int munmap(void *addr, size_t len);
```

**`munmap`** è la funzione che serve per liberare una porzione di memoria che era stata mappata precedentemente usando `mmap` (l’equivalente di `free` per la memoria allocata con `malloc`)