---
Created: 2025-05-03
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## Index
- [[#Segnali|Segnali]]
	- [[#Segnali#$ verb signal.h $|signal.h]]
- [[#Gestione dei segnali|Gestione dei segnali]]
	- [[#Gestione dei segnali#Default handler|Default handler]]
	- [[#Gestione dei segnali#Esecuzione handler|Esecuzione handler]]
- [[#Syscalls|Syscalls]]
	- [[#Syscalls#$ verb sigprocmask $|sigprocmask]]
	- [[#Syscalls#$ verb signal $|signal]]
	- [[#Syscalls#$ verb sigaction $|sigaction]]
	- [[#Syscalls#$ verb kill $|kill]]
	- [[#Syscalls#$ verb sigsuspend $|sigsuspend]]
---
## Segnali
I segnali sono degli interrupt software inviati dal kernel ad un processo oppure da un processo ad un altro processo

I segnali vengono generati da **condizioni anomale** per le quali il kernel invia un segnale ai processi interessati

>[!example] Condizioni anomale che generano segnali
>- richiesta di sospensione da parte dell’utente → `CTRL-Z=SIGSTOP`
>- richiesta di terminazione da parte dell’utente → `CTRL-C=SIGINT`
>- eccezione hardware → divisione per $0$, riferimento di memoria non valido, etc.

Ma un segnale può essere generato anche da condizioni non anomale

>[!example] Condizioni non anomale che generano segnali
>- terminazione di un processo figlio
>- un timer settato con `alarm` scade
>- un segnale inviato da un processo ad un altro processo mediante la syscall `kill`
>- un dato arriva su di una connessione di rete → `SIGURG`
>- un processo scrive su di una PIPE che non ha un “lettore” → `SIGPIPE`

### $\verb|signal.h|$
La lista completa dei segnali (costanti intere) è contenuta in `<signal.h>` e ad ogni evento è associato un segnale

>[!info] Segnali principali (`man 7 signal`)
>- `SIGINT 2` → terminazione, `CTRL+C` da tastiera
>- `SIGQUIT 3` → core dump, uscita
>- `SIGILL 4` → core dump, istruzione illegale
>- `SIGABR 6` → core dump, abort
>- `SIGFPE 8` → core dump, eccezione di tipo aritmetico
>- `SIGKILL 9` → terminazione, kill (non gestibile)
>- `SIGUSR1 10` → terminazione, definito dall’utente
>- `SIGSEGV 11` → core dump, segmentation Fault
>- `SIGUSR2 12` → terminazione, definito dall’utente
>- `SIGPIPE 13` → terminazione, scrittura senza lettori su pipe o socket
>- `SIGALRM 14` → terminazione, allarme temporizzato
>- `SIGTERM 15` → terminazione, terminazione software
>- `SIGCHLD 17` → ignorato, status del figlio cambiato
>- `SIGSTOP 19` → stop, sospende del processo (non gestibile)
>- `SIGTSTP 20` → stop, stop da tastiera
>- `SIGTTIN 21` → stop, lettura su `tty` in background
>- `SIGTTOU 22` → stop, scrittura su `tty` in background

---
## Gestione dei segnali
I segnali sono un esempio di **eventi asincroni**, ovvero eventi che possono avvenire in qualunque momento (non prevedibile dal programma) e al cui ricevimento il processo deve dire al kernel cosa fare

In generale si possono eseguire tre tipi di azione:
- ignorare il segnale → possibile con tutti tranne che con `SIGKILL` e `SIGSTOP`
- catturare il segnale (*catch*) → il processo chiede al kernel di eseguire una funzione definita dal programmatore (**signal handler**); i segnali `SIGKILL` e `SIGSTOP` non possono essere catturati
- eseguire l’azione di default → ad ogni segnale è associata un’azione di default (*default handler*)

### Default handler
Ogni segnale ha un default handeler. Eccone alcuni esempi:
- termina il processo e genera il core dump (file core) → salva sul disco lo stato della memoria del processo
- termina il processo senza generare il core dump
- ignora e rimuovi il segnale
- sospende il processo
- riesuma il processo

### Esecuzione handler
Quando un processo riceve un segnale che deve gestire con un handler:
1. interrompe il proprio flusso di esecuzione
2. esegue l’handler associato al segnale
3. riprende l’esecuzione dal punto in cui era stato interrotto

Tali passi possono essere realizzati con:
```c
int sigprocmask(int how, const sigset_t *set, sigset_t *oldset);
int rt_sigprocmask(int how, const kernel_sigset_t *set, kernel_sigset_t *oldset, size_t sigsetsize);
```

---
## Syscalls
### $\verb|sigprocmask|$

```c
int sigprocmask(int how, const sigset_t *set, sigset_t *oldset);
```

Consente di ottenere/settare la maschera segnali (ci dice i segnali bloccati)
Un segnale bloccato viene considerato pending e consegnato al processo solo quando viene sbloccato (ad eccezione di `SIGCHLD`, se più figli terminano mentre `SIGCHL` è bloccato il kernel non tiene traccia di tutte le singole istanze)

Potrebbe risultare utile ad esempio quando si sta eseguendo un’operazione delicata, e quindi viene ritardato l’arrivo del segnale al momento in cui l’operazione termina

L’argomento `how` dice come gestire il segnale e può assumere i seguenti valori:
- `SIG_BLOCK` → blocca i segnali definiti in set
- `SIG_UNBLOC` → sblocca i segnali definiti in set
- `SIG_SETMASK` → setta la maschera a set

`set` è la maschera da usare mentre in `old_set` viene salvata la maschera prima della modifica (può essere utile per ripristinare la maschera dopo la chiamata)

>[!hint]
>E’ importante ricordare che:
>- `sigprocmask` è applicabile solo a processi e non a threads (infatti ognuno ha la sua maschera)
>- `SIGKILL` e `SIGSTOP` non possono essere bloccati
>- ogni processo creato tramite `fork` eredita maschera dei segnali
>- la maschera dei segnali viene mantenuta dopo `exec`

### $\verb|signal|$

```c
sighandler_t signal(int signum, sighandler_t handler);
```

La syscall `signal` importa l’handler del segnale `signum` alla funzione hander passata come parametro. Restituisce `SIG_ERR` o il valore del precedente handler

Sono definite 2 macro:
- `SIG_IGN` → ignora il segnale
- `SIG_DFL` → assegna l’handler di default (utile per ripristinare l’handler del segnale a default)

>[!warning]
>L’uso di `signal` è **deprecato** perché l’implementazione non è standard e può variare da sistema a sistema

### $\verb|sigaction|$

```c
int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact);
```
- `signum` → segnale
- `act` → handler del segnale (se `NULL` viene usato `oldact`)
- `oldact` → handler precedente (viene popolato quando si invoca la syscall)

Questa è la struttura dell’`act`
```c
struct sigaction {
	// puntatore alla funzione signal handler
	// può essere SIG_IGN, SIG_DFL o puntatore a funzione
	void (*sa_handler)(int);
	
	// Alternativo a sa_handler
	void (*sa_sigaction)(int, siginfo_t *, void *);
	
	// speficia la maschera dei segnali che dovrebbero essere bloccati 
	// durante l’esecuzione dell’handler
	sigset_t sa_mask;
	
	// flags per modificare il comportamento del segnale
	int sa_flags;
	
	// obsoleto
	void (*sa_restorer)(void);
};
```

### $\verb|kill|$

```c
 int kill(pid_t pid, int sig);
```

La system call kill invia il segnale `sig` ad un processo `pid`:
- `pid>0` → il segnale è inviato al processo con pid `pid`
- `pid=0` → il segnale è inviato a tutti i processi del process group del chiamante
- `pid=-1` → il segnale è inviato a tutti i processi (tranne `init`) per cui il chiamante ha i privilegi per inviare un segnale
- `pid<-1` → il segnale è inviato a tutti i processi del process group del chiamante con pid uguale a `-pid`

### $\verb|sigsuspend|$

```c
int sigsuspend(const sigset_t *mask);
```

Sospende il processo che invoca la call e rimpiazza la maschera con mask. Il processo rimane sospeso finché non arriva un segnale per cui è definito un handler o un segnale di terminazione