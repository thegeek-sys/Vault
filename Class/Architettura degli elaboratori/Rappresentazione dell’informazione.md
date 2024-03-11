---
Created: 2024-03-11
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---

>[!info] Index
>- [[#Introduction|Introduction]]
>- [[#R-type|R-type]]
>	- [[#R-type#Esempio|Esempio]]
>- [[#I-type|I-type]]
>	- [[#I-type#Esempio|Esempio]]
>- [[#J-type|J-type]]

---
## Introduction
Siamo quindi pronti ad esaminare come viene interpretata un comando dato in linguaggio assembly in linguaggio macchina

![[Screenshot 2024-03-08 alle 11.36.00.png]]

> [!warning]
> Il linguaggio macchina potrebbe causare a confusione poiché in questo caso `rs` e `rd` sono invertiti

---
## R-type
Ogni istruzione è composta da 32 bit ed è divisa in cinque **campi**:
- *codeop* → operazione base dell’istruzione
- *rs* → registro contenente il primo operando sorgente
- *rt* → registro contenente il secondo operando sorgente
- *rd* → registro destinazione
- *shamt* → numero di posizioni di scorrimento (utilizzato solo per operazioni di shifting, default zero)
- *funct* → specifica la variante dell’operazione base definita dal codice operativo

![[Screenshot 2024-03-11 alle 19.10.12.png]]

Questo tipo di istruzioni:
- senza accesso alla memoria
- eseguono istruzioni aritmetico/logiche

### Esempio
Quindi se faccio `add $t0,$s1,$s2`
![[Screenshot 2024-03-08 alle 11.27.03.png]]
`rd` corrisponderà a t0
`rs` corrisponderà a s1
`rt` corrisponderà a s2


---
## I-type
Potrebbe però nascere un problema quando un’istruzione richiede campi di dimensioni maggiori rispetto a quelle delle istruzioni R-type. Ciò può avvenire ad esempio nell’istruzione di *load word* che richiede due registri e una costante la quale, per come abbiamo gestito lo spazio fino ad ora, non può superare il valore di $2^5$.
Per risolvere questo conflitto dunque è stato introdotto un altro tipo di istruzione: la **I-type** (immediato) che ha una differente predisposizione dei bit. In particolare vengono lasciati 16 bit per un indirizzo di memoria (o meglio una sua parte) o una costante.

![[Screenshot 2024-03-11 alle 19.19.07.png]]

Questo tipo di istruzioni:
- load/store
- salti condizionati (salto relativo al PC)

### Esempio
Quindi se faccio `addi $t2,$s2,4`
![[Screenshot 2024-03-08 alle 11.33.08.png]]

---
## J-type
Queste operazioni sono divise in:
- opcode → 6 bit
- indirizzo → 26
Eseguono operazioni di **salti non condizionati** (salto assoluto)