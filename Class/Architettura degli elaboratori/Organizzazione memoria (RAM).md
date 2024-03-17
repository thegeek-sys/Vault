---
Created: 2024-03-11
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
![[Screenshot 2024-03-11 alle 20.02.54.png]]

La memoria è organizzata come segue:
- I primi 4 byte sono riservati al kernel
- Poi abbiamo il programma dell’utente (anche chiamato `.text`) (il mio codice)
- In seguito i dati statici (`.data`) allocati in assegnamento
- Infine ho lo spazio libero in cui metto i dati dinamici e lo stack

Il **Global Pointer** ($gp) indica fino a dove interpretare lo spazio libero come dati dinamici non-locali
Lo **Static Pointer** ($sp) indica fino a dove interpretare lo spazio libero come stack (variabili locali, per le chiamate nidificate)
Il **Program Counter** viene utilizzato dalla CPU per tenere traccia di dove ci troviamo. Ogni volta che viene una istruzione viene letta il PC viene incrementato di 4 byte

