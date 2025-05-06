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

![[Pasted image 20250506215825.png|center|250]]

Quando il processo padre invoca le 2 `pipe` per creare un canale di comunicazione bidirezionale e fa una `fork`, il processo figlio eredita tutti e 4 i file descriptor (2 `pipe` ognuna delle quali crea 2 `fd`, uno per la lettura e uno per la scrittura), in questo modo entrambi i processi possono leggere e scrivere

Stato delle connessioni alle `pipe1` e `pipe2` dopo la `fork`:
![[Pasted image 20250506215731.png|450]]

In questo modo però, se un processo legge e scrive dalla stessa pipe, rischia di leggere i propri dati (non si ha una vera comunicazione), per questo motivo è necessario limitare la scrittura e la lettura del processo padre e figlio:
- il padre scrive in `pipe1` e il figlio legge da `pipe1` → viene chiusa la lettura su `pipe1` e la scrittura su `pipe2` per il padre
- il figlio scrive in `pipe2` e il padre legge da `pipe2`→ viene chiusa la lettura su `pipe2` e la scrittura su `pipe1` per il figlio

Stato delle connessioni alle `pipe1` e `pipe2` dopo la chiusura appropriata dei canali di read e write:
![[Pasted image 20250506215949.png|450]]
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