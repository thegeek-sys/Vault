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

