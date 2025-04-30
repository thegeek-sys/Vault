---
Created: 2025-04-29
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## La creazione $\verb|fork()|$
`Init` è il processo $0$ (`pid=1`) padre di tutti i processi del sistema in esecuzione. Da esso vengono creati tutti i processi mediante system call `fork()`

La creazione consiste nella duplicazione del processo per creare relazioni $\text{padre}\to \text{figlio}$
Tramite il comando `pstree 1` è possibile visualizzare l’albero (genealogico) dei processi

---
## Nascita e morte
![[Pasted image 20250429011745.png]]

Ogni processo nella gerarchia fa riferimento al processo padre, ovvero quando nasce eredita codice e parte dello stato dal processo padre invece quando termina/muore ritorna l’exit status al processo padre (nel caso in cui il processo padre termini/muoia prima del processo figlio, quest’ultimo è adottato da `init`)

---
## Zombie
![[Pasted image 20250429011931.png|300]]

Un processo zombie è un processo terminato, ma il cui PCB è mantenuto nella Process Table dal kernel per dare modo al processo padre di leggere l’exit status di tale processo. Se il padre di un processo muore/termina il processo rimane in stato zombie

---
## System call controllo processi
![[Pasted image 20250429012027.png]]
![[Pasted image 20250429012044.png]]

Richiedono:
```c
#include <sys/types.h>
#include <unistd.h>
```

---
## Ereditarietà attributi
**Non ereditati**:
- process id (pid) → il figlio ha il suo proprio pid
- parent pid (ppid) → nel figlio il parent pid è uguale al pid del padre
- timer → ogni processo ha i propri timer
- record lock/memory lock → due processi non possono detenere gli stessi lock
- contatori risorse → i contatori dell’utilizzo delle risorse sono impostati a zero nel figlio
- coda dei segnali → la coda dei segnali in attesa viene svuotata nel figlio

**Ereditati**:
- real ed effective user e group ID
- groups id
- working directory
- ambiente del processo
- descrittori dei file
- terminale di controllo
- memoria condivisa

---
## $\verb|fork()|$

```c
pid_t fork(void);
```

La syscall `fork` crea un nuovo processo che è la copia del processo chiamante, a parte alcune strutture dati come il PID

Una volta chiamata `fork()`, seppure eseguita una sola volta, ritorna due volte: una volta al processo che l’ha invocata, un’altra al nuovo processo che è stato generato dall’esecuzione della fork stessa
In caso di errore ritorna $-1$ al chiamante e non viene creato nessun processo figlio

E’ inoltre importante ricordare che quando viene forkato un processo le variabili globali vengono duplicate (dal padre al figlio) così come le variabili locali che vengono copiate sullo stack. Dunque le variabili del padre e del figlio non si influenzeranno (non sono condivise) 

---
## $\verb|exit()|$

```c
void _exit(int status); // unistd.h
void exit(int status); // stdlib.h
```
La **syscall** `_exit()` termina direttamente il processo che la invoca senza invocare handler. Con la terminazione:
- vengono chiusi tutti i file descriptor
- i child vengono ereditati dal processo $1$ (`child`)
- invia il segnale SIGCHLD al processo padre
- ritorna `status` e l’exit status al processo padre

La **funzione di libreria** `exit()`:
- invoca tutti gli handler registrati con `atexit` e `on_exit`
- chiude tutti i file descriptor, svuota gli stream `stdio` e li chiude
- termina il processo
- ritorna (`status & 0377`) al padre (vedi `wait()`)
- `EXIT_SUCCESS`e `EXIT_FAILURE` sono $2$ costanti predefinite che possono essere passate come status (soluzione portabile)

La differenza principale sta nel fatto che la syscall non flusha i buffer (`stdout` e `stderr`), quindi potrebbe accadere che se ad esempio viene un `fprintf` su `stderr` oppure un `printf` potrebbe accadere che gli output non verranno mai mostrati se bufferizzati

---
## $\verb|abort()|$

```c
void abort(void); // stdlib.h
```
La syscall `abort` serve per terminare bruscamente un processo in modo anomalo, simulando un crash. In particolare:
1. invia al processo il segnale `SIGABORT`
2. il processo termina subito, saltando qualsiasi tipo di cleanup (es. `free()`, `fclose()`, …)
3. può essere intercettato

---
## Come un programma C è lanciato e terminato
![[Pasted image 20250429213742.png]]

---
## $\verb|wait()|$, $\verb|waitpid|$
Queste chiamate di sistema si usano per:
- attendere cambiamenti di stato in un figlio del processo chiamante
- ottenere informazioni sul figlio in cui è avvenuto il cambiamento

Un cambiamento avviene quando:
- il processo figlio è terminato
- il figlio è stato arrestato da un segnale
- il figlio è stato ripristinato da un sengale

Se un figlio è terminato `wait`/`waitpid` permettono al sistema di rilasciare le risorse associate al figlio, infatti se non viene eseguita un’attesa allora il figlio terminato rimane in uno stato zombie (permettono al padre di recuperare lo stato di uscita del figlio)

Se un figlio ha già cambiato stato, allora le chiamate tornano immediatamente, altrimenti esse si bloccano fino a quando un figlio cambia stato o un gestore di segnale interrompe la chiamata

Infatti se il processo padre non chiamasse mai `wait` o `waitpid` il processo figlio, dopo la terminazione, rimarrebbe per sempre nello stato zombie (o finché non termina anche il processo padre)

```c
pid_t wait(int *status);
```
La syscall `wait()` sospende l’esecuzione del processo chiamante fino a quando uno dei suoi figli termina. Ritorna $-1$ in caso di errore

```c
pid_t waitpid(pid_t pid, int *status, int options);
```
La syscall `waitpid` sospende l’esecuzione del processo chiamante fino a quando un figlio specificato dall’argomento pid ha cambiato stato

Il valore di PID può essere:
- $<-1$ → attesa di q1ualunque processo figlio il cui group ID del processo sia uguale al valore assoluto di `pid` (utile in caso di modifica del group ID dei figli)
- $-1$ → aspettare qualunque processo figlio (equivale a `wait`)
- $0$ → aspettare qualunque processo figlio il cui group ID del processo sia uguale a quello del processo chiamante
- $>0$ → aspettare il figlio il cui ID di processo sia uguale al valore di `pid`

Il comportamento predefinito di `waitpid()` è attendere solo i figli terminati, ed è modificabile attraverso l’argomento `options`:
- `WNOHANG` → torna immediatamente se nessun figlio è uscito
- `WUNTRACED` → torna anche se un figlio si è arrestato (ma non tracciato attraverso `ptrace(2)`). Lo stato del figlio non tracciato che è stato arrestato è fornito anche se l’opzione non è specificata
- `WCONTINUED` → torna anche se un figlio arrestato è stato riesumato inviando `SIGCONT`

Queste opzioni possono essere messi in OR

### Verifica dello stato ritornato
`wait` e `waitpid` memorizzano in `status` `NULL` o il valore dello stato del processo figlio.

`status` può essere verificato con le seguenti macro:
- `WIFEXITED` → ritorna true se terminato normalmente
- `WEXITSTATUS` → ritorna l’exit status del figlio
- `WIFSIGNALED` → ritorna true se è terminato per la ricezione di un segnale
- `WTERMSIG` → ritorna il numero del segnale che ha causato la terminazione
- `WCOREDUMP` → ritorna true se ha generato un core dump
- `WIFSTOPPED` → ritorna true se il processo è in stato stopped per la ricezione di un segnale
- `WSTOPSIG` → ritorna il numero del segnale che ha causato lo switch in stato di stopped
- `WIFCONTINUED` → ritorna true se il processo ha continuato per la ricezione di un segnale `SIGCONT`

>[!example]
>```c
>...
>int &s;
>wait(&s);
>WIFIEXITED(&s); // o altra macro W*
>```

---
## Rimpiazzare un processo figlio con un altro processo
La syscall `fork()` crea 2 processi identici con attributi comuni e altri no, potrei quindi utilizzare il controllo del flusso per far si che il processo figlio faccia qualcosa diverso dal padre, ma risulta poco efficiente

Come soluzione è possibile approfittare della syscall `execve` o delle funzioni della famiglia `exec`, che permettono la sostituzione dell’immagine del processo che la invoca con una nuova immagine

```c
int execve(const char *filename, char *const argv[], char *const envp[]);
// funzioni di libreria
int execl(const char *path, const char *arg, ...);
int execlp(const char *file, const char *arg, ...);
int execle(const char *path, const char *arg, ..., char * const envp[]);
int execv(const char *path, char *const argv[]);
int execvp(const char *file, char *const argv[]);
int execvpe(const char *file, char *const argv[], char *const envp[]);
```

In particolare `execve` sostituisce l’immagine del processo con quella contenuta in `filename` (che è un file binario oppure uno script che inizia con `#! interpreter [opt-args]`)
L’argomento `argv[]` contiene gli argomenti in input per il comando eseguito (`argv[0]` deve essere il comando stesso) e l’array deve essere terminato con `NULL`
L’argomento `envp[]` contiene l’ambiente della nuova immagine (lo vediamo dopo)

`execve` sostituisce le zone di memoria `text`, `bss` e `stack` del processo che invoca la funzione con quelle del nuovo programma mandato in esecuzione

>[!example] `exec` esegue il programma `vi`
>![[Pasted image 20250430110659.png]]

### $\verb|execve|$ e gli attributi di processo
Quando si chiama al syscall `execve` alcuni attributi vengono persi mentre altri preservati
#### Preservati
- process ID (PID) e parent PID
- real uid e real gid
- groups id
- session ID
- terminale di controllo
- working directory e root directory
- maschera creazione file (umask)
- file locks
- file descriptors (ad eccezione di quelli con flag `FD_CLOEXEC` impostato)
- maschera dei segnali
- segnali in attesa

#### Non preservati
- effective uid ed effective gid
- memory mapping
- timers
- memoria condivisa
- memory lock

### Funzioni di libreria $\verb|exec*|$
Le funzioni `execl`, `execlp` e `execle` prendono in ingresso un numero variabile di puntatori per specificare gli argomenti in ingresso alla nuova immagine del processo, un puntatore per ogni argomento

Le funzioni `execv`, `execvp` e `execvpe` prendono in ingresso un puntatore ad un vettore per specificare gli argomenti in ingresso alla nuova immagine, un vettore dove ogni elemento è un argomento in ingresso

Le funzioni `execlp`, `execvp` e `execvpe` nel caso il nome del file non contenga uno `/`, usano la variabile di ambiente `PATH` per cercare il file da utilizzare per la nuova immagine (le altre usano path relativo o assoluto specificato in `filename`)

Le funzioni ``