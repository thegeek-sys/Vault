---
Created: 2024-12-18
Class: "[[Basi di dati]]"
Related:
---
---
## Introduction
In sistemi di calcolo con un a sola CPU i programmi sono eseguiti concorrentemente in modo *interleaved* (interfogliato), quindi la CPU può:
- eseguire alcune istruzioni di un programma
- sospendere quel programma
- eseguire istruzioni di altri programmi
- ritornare ad eseguire istruzioni del primo

Questo tipo di esecuzione è detta concorrente e permette un uso efficiente della CPU

### Accesso concorrente alla BD
In un DBMS la principale risorsa a cui tutti i programmi accedono in modo concorrente è la **base di dati**. Se sulla BD vengono effettuate solo letture (la BD non viene mai modificata), l’accesso concorrente non crea problemi. Se sulla BD vengono effettuate anche scritture (la BD viene modificata),