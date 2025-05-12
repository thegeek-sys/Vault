---
Created: 2025-05-12
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## Applicazioni multithread
In una applicazione tradizionale, il programmatore definisce un unico flusso di esecuzione delle istruzioni. La CPU esegue istruzioni macchina in sequenza e il flusso di esecuzione “segue” la logica del programma (cicli, funzioni, chiamate di sistema, gestori di segnali…)
Quando il flusso di esecuzione arriva ad eseguire la API `exit()` l’applicazione termina

Le applicazioni multithread consentono al programmatore di definire diversi flusso di esecuzione:
- ciascun flusso di esecuzione condivide le strutture dati principali dell’applicazione
- ciascun flusso di esecuzione procede in modo concorrente ed indipendente dagli altri flussi
- l’applicazione finisce solo quando tutti i flussi di esecuzione vengono terminati

---
## Processi e threads