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
Il `reader` invece