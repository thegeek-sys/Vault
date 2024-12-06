---
Created: 2024-12-06
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Introduction
Nel problema dei lettori/scrittori si ha un’**area dati condivisa** tra molti processi di cui **alcuni la leggono, altri la scrivono**

Condizioni da soddisfare:
- più lettori possono leggere l’area contemporaneamente (nei produttori/consumatori non era permesso)
- solo uno scrittore può scrivere nell’area
- se uno scrittore è all’opera sull’area, nessun lettore può effettuare letture

La vera grande differenza con i produttori/consumatori sta nel fatto che l’area condivisa **si accede per intero** (niente problemi di buffer pieno o vuoto, ma è importante permettere ai lettori di accedere contemporaneamente)

---
## Soluzione con precedenza ai lettori
![[Pasted image 20241206232642.png|center|400]]

Il `writer` ha come unico compito quello di scrivere e lo fa attraverso un semaforo di mutua esclusione (solo un `writer` per volta può scirvere)
Il `reader` invece (come per la soluzione di [[Semafori#Trastevere|trastevere]]) incrementa il valore di `readcount` e se si tratta del primo reader, vengono bloccati eventuali scrittori (ma se uno scrittore si trova già nella sezione critica è il lettore a bloccarsi). Viene quindi eseguita l’operazione di lettura e infine viene decrementato il valore di `readcount` e se si tratta dell’ultimo lettore, vengono sbloccati eventuali `writer`

Potrebbe accadere in questa soluzione che si vada in starvation sui `writer`

---
## Soluzione con precedenza agli scrittori
![[Screenshot 2024-12-06 alle 23.35.16.png]]

In questo caso ho fatto ciò che prima avevo fatto solo per i `reader`, per i `writer`
Per i `reader` invece è stato aggiunto, oltre al solito semaforo locale, il semaforo `rsem` che permette, essendo `writer` controllato da un semaforo locale, di evitare che il lettore possa prevalere sullo scrittore; infatti se ad esempio alla fine ad un certo punto arriva una 