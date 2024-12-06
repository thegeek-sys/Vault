---
Created: 2024-12-06
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Introduction
Le soluzione software è usabile solo per problemi di concorrenza semplici come ad esempio la mutua esclusione.
In questo caso dunque non possiamo fare affidamento alle istruzioni macchina, ma solamente all’assegnamento a variabili e simili. Come contro però si ha il fatto che tutte le operazioni sono in attesa attiva (non possono essere bloccati i processi)

---
## Primo tentativo
![[Pasted image 20241206174629.png]]
Questa soluzione però è applicabile solo a 2 processi (non a più processi)
### Problemi
Una soluzione come questa risolve il problema della mutua esclusione ma con dei problemi.
Il maggiore di tutti sta nel fatto che funziona se ci sono due processi, ma se ce ne fosse uno solo (`PROCESS 1`) e `turn` fosse inizializzato a $1$ non si uscirebbe mai del processo

---
## Secondo tentativo
![[Pasted image 20241206175202.png]]
Il `PROCESS 0` legge la variabile di `PROCESS 1` e scrive la propria, mentre `PROCESS 1` fa il contrario

### Problemi
In questo caso se lo scheduler interrompe `P0` immediatamente prima della modifica di `flag[0]`