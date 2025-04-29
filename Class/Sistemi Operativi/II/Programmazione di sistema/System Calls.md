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
## $\verb|fork()|$

```c
pid_t fork(void);
```

La syscall `fork` crea un nuovo processo che è la copia del processo chiamante, a parte alcune strutture dati come il PID

Una volta chiamata `fork()`, seppure eseguita una sola volta, ritorna due volte: una volta al processo che l’ha invocata, un’altra al nuovo processo che è stato generato dall’esecuzione della fork stessa
In caso di errore ritorna $-1$ al chiamante e non viene creato nessun processo figlio

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

---
## Come un programma C è lanciato e terminato
![[Pasted image 20250429213742.png]]
