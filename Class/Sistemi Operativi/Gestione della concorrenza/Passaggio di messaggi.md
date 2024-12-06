---
Created: 2024-12-06
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
---
---
## Introduction
Fino ad adesso abbiamo visto che la comunicazione tra i processi veniva fatta attraverso l’utilizzo di variabili globali, adesso utilizzeremo la comunicazione diretta, tramite la quale un processo può comunicare attraverso un messaggio ad un altro processo

---
## Interazione tra processi
Per l’interazione tra due processi devono essere soddisfatti due requisiti:
- sincronizzazione (mutua esclusione)
- comunicazione

Lo scambio di messaggi (*message passing*) è una soluzione al secondo requisito (funziona sia con memoria condivisa che distribuita)
Mentre per i semafori avevamo `wait` e `signal`, qui si hanno due istruzioni fondamentali `send(destination, message)` e `receive(source, message)` (`message` è un input per `send` mentre un output per `receive`) e spesso ci sta anche il test di recezione. Queste operazioni sono sempre **atomiche**

### Sincronizzazione
La comunicazione richiede anche la sincronizzazione tra processi (il mittente deve inviare prima che il ricevente riceva).
Dunque ora capiremo quali operazioni devono essere bloccanti oppure no (il test di recezione non è mai bloccanti)

---
## `send` e `receive` bloccanti
Se la `send` e la `receive` sono bloccanti vuol dire che un processo non può inviare un messaggio finché il precedente non è stato ricevuto e viceversa (chi prima fa l’operazione si blocca e chi la fa per seconda, oltre a non bloccarsi, sblocca anche la prima).
Tipicamente questo tipo di operazione viene chiamato *randevous* e richiede una sincronizzazione molto stretta

---
## `send` non bloccante
Più tipicamente come approccio si preferisce usare la `send` non bloccante (la indicheremo con `nbsend`) e la `receive` bloccante.
In questo caso succede che il mittente continua (non importa se il messaggio è stato ricevuto o no) mentre il ricevente, se è stato eseguito per secondo non si blocca, altrimenti si blocca finché non riceve un messaggio dal mittente

---
## `receive` non bloccante
Approccio piuttosto raro in cui accade che, indipendentemente se il messaggio è stato ricevuto o meno, l’esecuzione continua. Questa operazione viene indicata con `nbreceive` e può settare un bit nel messaggio per dire se la recezione è avvenuta oppure no. Se la recezione è non bloccante, allora tipicamente non lo è neanche l’invio

---
## Indirizzamento