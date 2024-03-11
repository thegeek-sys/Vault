---
Created: 2024-03-11
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Siamo quindi pronti ad esaminare come viene interpretata un comando dato in linguaggio assembly in linguaggio macchina

---
## R-type
Ogni istruzione è composta da 32 bit ed è divisa in cinque **campi**:
- *codeop* → operazione base dell’istruzione
- *rs* → registro contenente il primo operando sorgente
- *rt* → registro contenente il secondo operando sorgente
- *rd* → registro destinazione
- *shamt* → numero di posizioni di scorrimento (utilizzato solo per operazioni di shifting, default zero)
- *funct* → specifica la variante dell’operazione base definita dal codice operativo

