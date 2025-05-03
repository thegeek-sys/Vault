---
Created: 2025-05-03
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
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
