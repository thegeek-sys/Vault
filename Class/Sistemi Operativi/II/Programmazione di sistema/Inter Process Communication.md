---
Created: 2025-05-05
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## PIPE e FIFO
Unix mette a disposizione due tipi di **Inter Process Communication** (*IPC*): **fifo** (o *named pipe*), per far comunicare processi non imparentati, e **pipe** (o *unamed pipe*), per far comunicare processi con un antenato in comune

La scrittura dei dati su una pipe/fifo (e quindi anche la lettura) avviene in maniera sequenziale (first-in first-out)

La fifo è uno speciale tipo di file che può essere creato per mezzo delle system call `mkfifo` o `mknod` (può essere utilizzata da più processi)
La pipe invece è una struttura dati in memoria *half-duplex* (la comunicazione è unidirezionale, un processo scrive l’altro legge). La creazione della pipe può essere effettuata con la system call `pipe` che crea due file descriptor (uno in lettura e uno in scrittura). Ad esempio un processo crea una pipe, poi crea un figlio e usa la pipe per comunicare con il figlio che eredita i descrittori dei file

## Pipe
Le pipe sono usato per IPC e, come già detto, sono unidireziali in particolare tra un processo padre e un figlio

Dopo il `fork()`
### $\verb|pipe|$

```c
int pipe(int pipefd[2])
```

- `pipefd[0]` → file descriptor di input
- `pipefd[1]` → file descriptor di output

I dati scritti sulla pipe sono bufferizzati dal kernel finché non sono letti e le pipe hanno una dimensione massima definita dal sistema

Comportamento:
- se un processo legge da una pipe vuota allora rimane bloccato
- se un processo scrive su una pipe piena allora rimane bloccato
- una pipe viene chiusa quando tutti e due i processi hanno invocato la `close`
- operazioni di lettura (`read`) su una pipe il cui `fd` di scrittura è stato chiuso con `close` ritorna $0$
- operazioni di scrittura (`write`) su una pipe il cui `fd` di lettura è stato chiuso con `close` ritornano $-1$ e ricevono il segnale `SIGPIPE`