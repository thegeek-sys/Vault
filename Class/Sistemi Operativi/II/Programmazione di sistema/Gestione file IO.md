---
Created: 2025-04-28
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## Index
- [[#Operare sui file|Operare sui file]]
- [[#File descriptor|File descriptor]]
	- [[#File descriptor#File flags associati a file descriptor|File flags associati a file descriptor]]
		- [[#File flags associati a file descriptor#Rappresentazione dei flag|Rappresentazione dei flag]]
		- [[#File flags associati a file descriptor#File status flags|File status flags]]
- [[#System call principali|System call principali]]
	- [[#System call principali#$ verb open() $|open()]]
		- [[#System call principali#$ verb open() $#Differenza con $ verb fopen $|Differenza con fopen]]
		- [[#System call principali#$ verb open() $#Flags|Flags]]
	- [[#System call principali#$ verb read() $]]
		- [[#System call principali#Differenza con $ verb fread $|Differenza con fread]]
	- [[#System call principali#$ verb write() $|write()]]
	- [[#System call principali#$ verb close() $|close()]]
	- [[#System call principali#Altre syscall e funzioni di libreria|Altre syscall e funzioni di libreria]]
		- [[#Altre syscall e funzioni di libreria#$ verb dup() $|dup()]]
		- [[#Altre syscall e funzioni di libreria#$ verb stat() $|stat()]]
		- [[#Altre syscall e funzioni di libreria#$ verb fstat() $|fstat()]]
		- [[#Altre syscall e funzioni di libreria#$ verb chmod() $ e $ verb fchmod() $|chmod() e fchmod()]]
		- [[#Altre syscall e funzioni di libreria#$ verb opendir() $, $ verb readdir() $ e $ verb closedir() $|opendir(), readdir() e closedir()]]
		- [[#Altre syscall e funzioni di libreria#$ verb fcntl() $|fcntl()]]
		- [[#Altre syscall e funzioni di libreria#$ verb select() $|select()]]
---
## Operare sui file

>[!info]
>Per file si intende una astrazione che descrive ogni risorsa (es. file su disco, dispositivi, socket, fifo)

Sequenza tipica di azione su un file:
1. apertura file mediante `open` → viene creato un file descriptor
2. operazione su file mediante syscall → es. `write`, `read`
3. chiusura del file mediante `close`

Quando un processo termina, chiude automaticamente tutti i file correntemente aperti, tuttavia, è sempre consigliabile e buona regola di programmazione chiudere esplicitamente un file invocando la `close`

---
## File descriptor
Un **file descriptor** è il riferimento ad un file aperto ed è rappresentato da un intero piccolo, generato sequenzialmente a partire dal valore $0$.

![[Pasted image 20250428214601.png|400]]

Per default, ogni processo, ha associato i seguenti file descriptor:
- $0$ → `stdin`
- $1$ → `stdout`
- $2$ → `stderr`

Quando un file viene chiuso, il suo file descriptor viene liberato e può essere riutilizzato. E’ inoltre possibile aprire uno stesso file ed ottenere file descriptor diversi che puntano allo stesso file

### File flags associati a file descriptor
Si differenziano due tipi di flags:
- **file status flags** → associati allo stato del file e condivisi tra tutti i file descriptor ottenuti per duplicazione da un unico file descriptor
- **file descriptor flags** →
	- associati al file descriptor (non al file aperto)
	- indipendenti dal contenuto o status del file (file descriptor diversi che fanno riferimento ad uno stesso file hanno ognuno il proprio insieme di descriptor flags)
	- descrivono proprietà e comportamento delle operazioni effettuate sul file
	- alcuni definiti solo per alcuni tipi di file speciali (es. fifo, socket)

#### Rappresentazione dei flag
I flag sono rappresentati con maschere di bit. Per poter gestire i flag si usano dei bit (ciascun bit rappresenta un flag). Quindi:
- `1` → flag attivo
- `0` → flag non attivo

Per ogni flag esiste una maschera (valore numerico dove un certo bit è $1$ e tutti gli altri $0$). Ad esempio `O_RDONLY` è una maschera
Se si vogliono attivare più flag insieme è possibile fare un ora bitwise

```
MACRO1 = 01000000;
MACRO2 = 00010000;
MACRO1|MACRO2 = 01010000;
```

#### File status flags
Esistono tre categorie di file status flags:
- *modalità di accesso* → `read`, `write`, `read&write` che sono specificati nella `open` e non possono essere modificati una volta aperto il file
- *di apertura* → definiscono il comportamento della open e non vengono mantenuti
- *modalità operative* → definiscono il comportamento delle operazioni `read` e `write` e sono specificati nella open ma possono essere modificati anche dopo l’apertura del file

---
## System call principali
- `open`
- `chown`
- `read`
- `chmod`
- `write`
- `stat`
- `lseek`
- `select`
- `close`
- `ioctl`
- `unlink`
- `fnctl`
- `symlink`
- `rename`
- `rmdir`
- `chdir`

### $\verb|open()|$

```c
int open(const char *pathname, int flags);
int open(const char *pathname, int flags, mode_t mode);
```

La syscall `open` restituisce $-1$ in caso di errore, altrimenti il file descriptor
Il parametro `flags` (file status flags) è l’equivalente del parametro `mode` della `fopen`
Infine il parametro opzionale `mode` indica i bit dei permessi di un file in creazione

![[Pasted image 20250428221217.png|450]]

#### Differenza con $\verb|fopen|$
Mentre `open` ritorna solo il file descriptor `fopen` ritorna il puntatore a un oggetto `FILE`. Questo oggetto è tipicamente una struttura che contiene tutte le informazioni richiesta dalla standard libreria I/O per gestire lo stream, in particolare contiene:
- il file descriptor al file effettivo
- un puntatore ad un buffer per lo stream
- la dimensione del buffer
- un conteggio del numero di caratteri attualmente contenuti nel buffer
- un flag di errore

#### Flags
Come già detto ci sono diversi tipi di flags:
- flags modalità di accesso
	- `O_RDONLY`
	- `O_WRONLY`
	- `O_RDWR`
- flags di apertura
	- `O_CREAT` → crea il file se non esiste
	- `O_EXCL` (quando specificato insieme a `O_CREAT` dà errore se il file esiste già)
	- …
- flags modalità operativa
	- `O_APPEND` → scrive sempre alla fine del file
	- `O_SYNC` → scrittura sincrona, la call ritorna solamente quando la scrittura dei dati nel file è terminata
	- `O_TRUNC` → se il file esiste, è un file regolare, e la modalità di accesso consente la scrittura, allora il file viene troncato alla posizione $0$

### $\verb|read()|$

```c
ssize_t read(int fd, void *buf, size_t count);
```

- `fd` → file descriptor
- `buf` → puntatore all’area di memoria in cui memorizzare i byte letti
- `count` → numero di byte da leggere

Restituisce $-1$ in caso di errore, altrimenti il numero di byte letti (che può essere minore di `count` se si raggiunge EOF)

#### Differenza con $\verb|fread|$
Mentre `fread` legge da uno stream di tipo `FILE` (bufferizzata) `read` non è bufferizzata. Inoltre `fread` prende in input la dimensione del tipo di dato da leggere, mentre `read` lavora sui byte indipendentemente dal tipo di dati in essi contenuto

### $\verb|write()|$

```c
ssize_t write(int fd, const void *buf, size_t count);
```

- `fd` → file descriptor
- `buf` → area di memoria da cui leggere i dati; dichiarata `const` per non essere modificata dalla funzione
- `count` → numero di byte da scrivere

Restituisce $-1$ in caso di errore, altrimenti il numero di byte scritti (che può essere minore di `count` se si ad esempio dà un segnale)

### $\verb|close()|$

```c
int close(int fd);
```

Permette di chiudere il file descriptor `fd` (il file descriptor viene liberato e può essere quindi riutilizzato)
Ritorna $-1$ in caso di errore e $0$ in caso la chiamata termini correttamente. Nel caso venga chiuso l’ultimo file descriptor che fa riferimento ad un file rimosso, allora il file viene cancellato

### Altre syscall e funzioni di libreria

```c
int dup(int oldfd);
int stat(const char *path, struct stat *buf);
int chmod(const char *path, mode_t mode)
int chown(const char *path, uid_t owner, gid_t group)
int rename(const char *oldpath, const char *newpath);
int mkdir(const char *pathname, mode_t mode);
DIR *opendir(const char *name); //Libreria
struct dirent *readdir(DIR *dirp); //Libreria
int closedir(DIR *dirp); //Libreria
int chdir(const char *path);
int fcntl(int fd, int cmd, ... /* arg */ );
int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout); // la vedremo dopo
```

#### $\verb|dup()|$
Duplica il file descriptor `oldfd` e restituisce il valore del nuovo `fd`. Restituisce $-1$ in caso di errore

#### $\verb|stat()|$
Restituisce informazioni di stato riguardo uno specifico file e le memorizza nell’area di memoria puntata da `buf`

![[Pasted image 20250429002302.png]]

Sono definire anche una serie di macro da utilizzare sulla struttura dati `stat` (buf) per verificare il tipo del file:
- `S_ISREG(m)`, `S_ISDIR(m)`, `S_ISCHR(m)`,
- `S_ISBLK(m)`, `S_ISFIFO(m)`, `S_ISLNK(m)`,
- `S_ISSOCK(m)`

#### $\verb|fstat()|$
Restituisce in `buf` le informazioni di stato del file specificato con nome file `path` o con file descriptor `fd`. Ritorna $0$ se termina correttamente, $-1$ altrimenti

#### $\verb|chmod()|$ e $\verb|fchmod()|$
La syscall `chmod` permette di cambiare il file mode. Ritorna $-1$ se errore, $0$ altrimenti
Il parametro `mode` è un numero ottale (es. $0755$). Si possono inoltre usare le maschere predefinite tipo:

![[Pasted image 20250429002835.png]]

#### $\verb|opendir()|$, $\verb|readdir()|$ e $\verb|closedir()|$

```c
DIR *opendir(const char *name);
struct dirent *readdir(DIR *dirp);
int closedir(DIR *dirp);
```

Non sono system call ma funzioni di libreria che permettono di gestire una directory tramite stream (ritornato da `opendir`). `readdir` legge il contenuto della directory (prossimo elemento disponibile) ritornando la struttura `dirent` o NULL se non ci sono più elementi

#### $\verb|fcntl()|$

```c
int fcntl(int fd, int cmd, ... /* arg */ );
```

E’ una system call che permette di effettuare operazioni sul file descriptor `fd`, ad esempio:
- duplicazione del `fd`
- manipolazione flag file descriptor
- manipolazione flag di stato
- gestione lock su `fd`

A tali insiemi di operazioni sono associati un insieme di comandi passati come parametro `cmd`, mentre gli argomenti del comando `cmd`, se presenti, si passano come parametri `arg` della syscall e vanno messi dopo `cmd`

Esempi:
```c
fcntl(fd, F_GETFL); // restituisce file access mode e file status flag
fcntl(fd, F_SETFL, O_APPEND); // importa i file status flag
ftncl(fd, F_SETLK, F_WRLCK); // acquisice/rilascia lock
ftncl(fd, F_SETLKW, F_WRLCK); // acquisisce/rilascia lock bloccante (se è presente un lock attende il suo rilascio)
fcntl(fd, F_GETLK, F_WRLCK); // testa se è presente un lock di scrittura, se il lock può essere messo fcntl aggiorna l_type a F_UNLCK. deve quindi essere eseguita da F_SETLK/LKW
```

L’argomento è la struttura lock:
- `l_whence` → dice da dove parte
- `l_start` (> o < 0)
	- `SEEK_SET`→ inizio file
	- `SEEK_CUR`→ posizione corrente
	- `SEEK_END`→ fine file
- `l_len=0` → tutto il file viene bloccato altrimenti vengono bloccati `l_len` byte a partire da `l_start`
- `F_SETLK` → acquisisce lock se `l_type=F_RDLCK` o `F_WRLCK`, rilascia se `l_type=F_UNLCK`; restituisce $-1$ se un altro processo ha il lock
- `F_SETLKW` → come la `F_SETLK` ma se c’è un lock sul file attende (`wait`) che il lock venga rilasciato
- F_GETLK, testa esistenza lock; la struttura `flock` contiene il lock che si vuole mettere, se il lock può essere messo `fcntl` aggiorna `l_type` a `F_UNLCK` (deve quindi essere seguita da `F_SETLK`/`LKW`)

Il lock è *advisory*, ovvero richiede cooperazione tra processi, infatti tutti I processi fanno una `F_GETLK` o `F_SETLK/LKW` e osservano il risultato (cercare di scrivere un file sul quale un processo detiene un lock non ha l'effetto di bloccare la scrittura)
Per avere un lock mandatory, ovvero che impedisce la scrittura o lettura richiede che il file system supporti il mandatory locking

#### $\verb|select()|$
La syscall `select` ha lo scopo di attendere che almeno uno tra i file descriptor `fd` diventi pronto per leggere, scrivere o segnalare errori

Parametri:
- `nfds` → il più grande `fd` + 1 tra tutti quelli nei set (`readfds`, `writefds`, etc.).
- `readfds`→ `fd` da controllare per **lettura** (es. `read()` non bloccante).
- `writefds`→ `fd` da controllare per **scrittura** (es. `write()` non bloccante).
- `exceptfds`→ `fd` da controllare per **eccezioni** (es. errori su socket).
- `timeout` → quanto tempo **aspettare** (può essere 0 o NULL).

Dopo l’esecuzione ritorna:
- $>0$ → numero di descriptor pronti
- $0$ → timer scaduto, nessun `fd` pronto
- $-1$ → errore

Inoltre `readfds`, `writefds`, `exceptfds` vengono modificati per contenere solo i fd effettivamente pronti. Quindi, dopo `select()`, è necessario usare `FD_ISSET(fd, &readfds)` per sapere se un `fd` è pronto.

Il parametro `timeout` controlla **quanto aspetta** `select()` prima di restituire:
- `NULL` (nessun timeout) → la chiamata è bloccante, quindi attende indefinitamente finché un `fd` non è pronto.
- `timeout = 0` → la chiamata è non bloccante, quindi controlla i `fd` e ritorna subito.
- `timeout > 0` → attende per quel tempo massimo

Sono messe a disposizioni quattro macro per la gestione degli insieme dei file descriptor:
- `FD_ZERO()` → svuota un insieme.
- `FD_SET()` → aggiunge un file descriptor ad un insieme
- `FD_CLR()` → rimuove un file descriptor da un insieme
- `FD_ISSET()` → testa se un file descriptor appartiene ad un insieme. Utile quando select ritorna

I prototipi sono:
```c
void FD_ZERO(fd_set *set);
void FD_SET(int fd, fd_set *set);
void FD_CLR(int fd, fd_set *set);
int FD_ISSET(int fd, fd_set *set);
```